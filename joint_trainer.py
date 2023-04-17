import os
import json
import cv2
import time
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from collections import Counter
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torchvision.ops import nms
import numpy as np
import math

# Split batch by dataset before training
def split_by_dataset(inputs, targets, dataset_indexes, num_datasets):
	input_targets_by_dataset = [{'inputs':[], 'targets':[]} for i in range(num_datasets)]
	for i in range(len(dataset_indexes)):                    
		dataset_index = dataset_indexes[i]
		input_targets_by_dataset[dataset_index]['inputs'].append(inputs[i])
		input_targets_by_dataset[dataset_index]['targets'].append(targets[i])
	return input_targets_by_dataset


# Images can be moved directly; targets must be moved differently depending on type
def move_to_gpu(inputs, targets):
	device = torch.device('cuda')
	inputs_gpu = []
	targets_gpu = []
	for i in range(len(inputs)):
		inputs_gpu.append(inputs[i].to(device))
		if isinstance(targets[i], int):
			targets_gpu.append(torch.tensor(targets[i]).to(device))
		elif isinstance(targets[i], dict):
			targets_gpu.append({k: v.to(device) for k, v in targets[i].items()})
		else:
			targets_gpu.append(targets[i].to(device))
			
	return inputs_gpu, targets_gpu
		
	
class JointTrainer():
	"""Trains with samples from every dataset
	
	"""
	def __init__(self):
		self.alpha = 0.75 # For weighting losses
	
	# Validate on each dataset separately
	# Keep track of max failures and end early if we're not going to send these weights anyway
	def validate(self, models, val_dataloaders, tasks, eval_methods, training_params):
		print('VALIDATING')
		num_models_to_test = len(models)
		order_to_validate = training_params['order_to_validate']
		print(f'Validation order: {order_to_validate}')
		num_to_meet = len(training_params['indexes_models_met']) + 1
		max_failures = len(models) - num_to_meet
		unmerged_accs = training_params['unmerged_accs']
		target_percentage = training_params['target_percentage']

		accs = [None for m in models]
		num_failed = 0
		for i in order_to_validate:
			if i in training_params['skip_list']:
				continue
			print(f'Dataset {i}:')
			model = models[i]
			model.eval()
			if tasks[i] == 'image_classification':
				ic_acc = eval_methods[i](model, val_dataloaders[i])
				accs[i] = ic_acc
				
			if tasks[i] == 'object_detection':
				od_acc = eval_methods[i](model, val_dataloaders[i])
				if od_acc:
					accs[i] = od_acc
				else:
					accs[i] = 0.0

			if accs[i] < (unmerged_accs[i]*target_percentage):
				num_failed += 1
				print(f'Failed. {num_failed} failures out of max {max_failures}')

			if num_failed > max_failures:
				break
				
		return accs

	# Fraction of batches is number of batches we should run on before validating again. Turn that into correct batch number
	def next_val_index(self, fraction_of_batches, idx, batches_per_epoch, epoch, num_epochs):
		val_batch_index = idx + math.ceil(batches_per_epoch * fraction_of_batches) - 1
		if val_batch_index > batches_per_epoch:
			if epoch < (num_epochs - 1):
				val_batch_index = val_batch_index%(batches_per_epoch)
			else:
				val_batch_index = batches_per_epoch - 1
		return val_batch_index


	def train(self, models, train_dataloader, val_dataloaders, tasks, unmerged_accs, eval_methods, dataset_names, results_path, lr=0.00001, num_epochs=10):
		# Initialize dict to keep track of training
		training_params = {'fraction_of_data': 1.0,
					'indexes_models_met': [],
					'order_to_validate': range(len(models)),
					'prev_accs': [],
					'all_accs': [],
					'skip_list': [],
					'unmerged_accs': unmerged_accs,
					'target_percentage': 0.95,
					'loss_weights': [1.0 for i in models],
					'start_time': time.time(),
					'indexes_over_time': [],
					'done': False}

		# Add all parameters to optimizer
		parameters_all = []
		for m, model in enumerate(models):
			model = model.to('cuda')
			parameters_all.extend(list(model.parameters()))
		optimizer = SGD(parameters_all, lr=lr, momentum=0.9, weight_decay=0.0005)

		# Start by calculating accuracy before doing any training
		training_params = self.validate_and_update_params(models, val_dataloaders, tasks, eval_methods, training_params, 0.0, results_path, dataset_names)
		if training_params['done']:
			return training_params['indexes_over_time']

		# Based on how close we are to accuracy, decide how much of the data to run on
		val_batch_index = self.next_val_index(training_params['fraction_of_data'], 0, len(train_dataloader), -1, num_epochs)
		print(f'Next val index: {val_batch_index}')

		# Start training
		for epoch in range(num_epochs):
			print(f'Epoch {epoch}/{num_epochs}')
			for model in models:
				model.train()
			# Iterate through batches
			for idx, (inputs_targets, dataset_indexes) in enumerate(train_dataloader):
				if idx == val_batch_index:
					# Get a more exact read on how far we are into training to help decide when to give up
					current_epoch = epoch + (idx/len(train_dataloader))
					training_params = self.validate_and_update_params(models, val_dataloaders, tasks, eval_methods, training_params, current_epoch, results_path, dataset_names)
					if training_params['done']:
						return training_params['indexes_over_time']
					val_batch_index = self.next_val_index(training_params['fraction_of_data'], idx, len(train_dataloader), epoch, num_epochs)
					print(f'Next val index: {val_batch_index}')

					for model in models:
						model.train()

				# Each batch is selected randomly among all data combined.
				# Separate each image/target by which dataset index they came from and train each separately, summing the losses
				num_datasets = len(Counter(dataset_indexes).keys())
				inputs_all, targets_all = move_to_gpu(inputs_targets[0], inputs_targets[1])
				inputs_target_by_datasets = split_by_dataset(inputs_all, targets_all, dataset_indexes, len(models))

				with torch.set_grad_enabled(True):
					losses = []
					for i in range(len(models)):
						if i in training_params['skip_list']:
							continue
						if not inputs_target_by_datasets[i]['inputs']: # no inputs for this dataset
							continue
						inputs_dataset = torch.stack(inputs_target_by_datasets[i]['inputs'])
						targets_dataset = inputs_target_by_datasets[i]['targets']
						if tasks[i] == 'object_detection':
							# model calculates loss by itself
							loss_dict = models[i](inputs_dataset, targets_dataset)
							od_loss = (sum(loss for loss in loss_dict.values()))
							if (idx < 20) or (idx%50 == 0):
								print(f'Iteration: {idx}/{len(train_dataloader)}, OD Loss (model index {i}): {od_loss}')
							losses.append(od_loss*training_params['loss_weights'][i])
						elif tasks[i] == 'image_classification':
							# model just gives predictions, use criterion to get loss
							criterion = nn.CrossEntropyLoss()
							targets_dataset = torch.stack(targets_dataset)
							outputs_dataset = models[i](inputs_dataset)
							_, preds = torch.max(outputs_dataset, 1)
							ic_loss = criterion(outputs_dataset, targets_dataset)
							if (idx < 20) or (idx%50 == 0):
								print(f'Iteration: {idx}/{len(train_dataloader)}, IC Loss (model index {i}): {ic_loss}')
							losses.append(ic_loss*training_params['loss_weights'][i])
						else:
							assert False, f'Unsupported task: {tasks[i]}'
					
					if not losses:
						continue
					loss = sum(losses)

					if math.isnan(loss):
						print('Loss became nan')
						return training_params['indexes_over_time']
					
					loss.backward()
					optimizer.step()
					optimizer.zero_grad()

		return training_params['indexes_over_time']


	def validate_and_update_params(self, models, val_dataloaders, tasks, eval_methods, training_params, current_epoch, results_path, dataset_names):
		current_accs = self.validate(models, val_dataloaders, tasks, eval_methods, training_params)
		updated_params = self.update_params(models, current_accs, training_params, current_epoch, results_path, dataset_names)
		return updated_params

	# Calculate how close we are from each acc target
	def distances_from_target(self, current_accs, target_accs, target_percentage, prev_accs):
		assert(len(current_accs) == len(target_accs))
		models_met = []
		fractions_of_data = []
		for i in range(len(current_accs)):
			current_acc = current_accs[i]
			if current_acc == None:
				current_acc = prev_accs[i]
			target_acc = target_accs[i] * target_percentage

			gap = max(0, (1 - (current_acc/target_acc)))
			print(f'gap: {gap}')
			if gap == 0:
				# Model has already met accuracy - make fraction = 1.0 because we only consider values <1
				models_met.append(i)
				fractions_of_data.append((i, 2.0))

			else:
				# Model has not met target yet
				# If this isn't the first acc calculation, normalize by lift since last time
				if prev_accs:
					lift = current_acc - prev_accs[i]
				else:
					print(f'No prev accs, lift is {current_acc}')
					lift = 0.1 # For the first round of training, use gap*10 as fraction
				
				if lift <= 0:
					fractions_of_data.append((i, 1.0))
				else:
					fractions_of_data.append((i, min(1.0, gap/lift)))

		# Organize result to help next round of validation
		print(f'Fractions: {fractions_of_data}')

		sorted_fractions = sorted(fractions_of_data, key = lambda x: x[1])
		print(f'Sorted: {sorted_fractions}')
		index_order_next_val = [f[0] for f in sorted_fractions]
		fractions_of_data_min = sorted_fractions[0][1]
		print(f'Min fraction: {fractions_of_data_min}')

		return fractions_of_data_min, index_order_next_val, models_met

	# Based on current accs, update training parameters
	def update_params(self, models, current_accs, training_params, current_epoch, results_path, dataset_names):
		num_models = len(training_params['unmerged_accs'])
		prev_accs = training_params['prev_accs']
		for i, acc in enumerate(current_accs):
			if acc == None:
				acc = prev_accs[i]
				current_accs[i] = acc
		print(f'Current accs: {current_accs}')

		updated_all_accs = training_params['all_accs']
		updated_all_accs.append((current_accs, current_epoch))

		# Check if we should eliminate any: hasn't improved in 2 epochs (ugly - fix)
		updated_skip_list = training_params['skip_list']
		if len(updated_all_accs) > 2:
			for i in range(num_models):
				not_improving = -1
				unmerged_acc = training_params['unmerged_accs'][i]*training_params['target_percentage']
				gap = max(0.0, unmerged_acc - updated_all_accs[-1][0][i])
				if gap > 0.0:
					lift = updated_all_accs[-1][0][i] - updated_all_accs[-2][0][i]
					epochs = updated_all_accs[-1][1] - updated_all_accs[-2][1]
					if ((10 - updated_all_accs[-1][1]) * lift)/epochs < gap:
						not_improving += 1

				# unmerged_acc = training_params['unmerged_accs']*training_params['target_percentage']
				gap2 = max(0.0, unmerged_acc - updated_all_accs[-2][0][i])
				if gap2 > 0.0:
					lift2 = updated_all_accs[-2][0][i] - updated_all_accs[-3][0][i]
					epochs2 = updated_all_accs[-2][1] - updated_all_accs[-3][1]
					if ((10 - updated_all_accs[-1][1]) * lift2)/epochs2 < gap2:
						not_improving += 1                

				if not_improving == 1 and (updated_all_accs[-1][1] - updated_all_accs[-3][1] >= 2):
					print(f'Adding model {i} to skip list')
					updated_skip_list.append(i)
			

		updated_weights = self.update_weight_tasks(current_accs, training_params['unmerged_accs'], training_params['target_percentage'], training_params['prev_accs'])
		fraction_of_data, order_to_validate, models_met = self.distances_from_target(current_accs, training_params['unmerged_accs'], training_params['target_percentage'], training_params['prev_accs'])
		done_training = len(models_met) >= num_models - len(updated_skip_list)
		should_send_indexes = ((len(models_met) > len(training_params['indexes_models_met'])) and (len(models_met) > 1))
		updated_indexes_times = training_params['indexes_over_time']

		# If we have made progress in memory savings since last time, send (save) these weights
		if should_send_indexes:
			curr_time = time.time() - training_params['start_time']
			updated_indexes_times.append((models_met, curr_time))
			for i, model in enumerate(models):
				if i in models_met:
					torch.save(model.state_dict(), os.path.join(results_path, 'weights', f'{dataset_names[i]}.pt'))


		updated_training_params = {'fraction_of_data': fraction_of_data,
								'indexes_models_met': models_met,
								'order_to_validate': order_to_validate,
								'prev_accs': current_accs,
								'all_accs': updated_all_accs,
								'skip_list': updated_skip_list,
								'unmerged_accs': training_params['unmerged_accs'],
								'target_percentage': training_params['target_percentage'],
								'loss_weights': updated_weights,
								'start_time': training_params['start_time'],
								'indexes_over_time': updated_indexes_times,
								'done': done_training}

		print(f'Should send: {should_send_indexes}, Done training: {done_training}')
		return updated_training_params

	# Use formula target/current ^ alpha
	def update_weight_tasks(self, current_accs, target_accs, target_percentage, prev_accs, max_weight=10.0):
		assert(len(current_accs) == len(target_accs))
		updated_weights = []

		for i in range(len(current_accs)):
			current_acc = current_accs[i]
			if current_acc == None:
				current_acc = prev_accs[i]
			target_acc = target_accs[i] * target_percentage
			if current_acc == 0.0:
				task_weight = max_weight
			else:
				task_weight = min(math.pow((target_acc/current_acc), self.alpha), max_weight)
			updated_weights.append(task_weight)
		print(f'Updated weights: {updated_weights}')
		return updated_weights


		  
