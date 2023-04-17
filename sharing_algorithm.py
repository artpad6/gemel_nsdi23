import os
import json
import torch
import time
import random
from collections import OrderedDict
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from mem_usage_summary import summary

# Base class for all sharing algs
class Sharing_Algorithm():
	def __init__(self, shared_layer_manager, models, path_for_storage):
		self.models = models
		self.path_for_storage = path_for_storage
		self.shared_layer_manager = shared_layer_manager
		self.layer_list = self.layer_instances_from_models(models)
		
		self.configs = self.generate_configs()
		self.current_layers = None
		self.config_index = None

	# Returns whether two layers will be counted as equal
	def layers_equal(self, layer1, layer2):
		layer1_components = layer1.split('), ')
		layer2_components = layer2.split('), ')
		if len(layer1) != len(layer2):
			return False
		stride_index = -1
		for i, component in enumerate(layer1_components):
			if 'stride' in component:
				stride_index = i

		same = True
		for j in range(len(layer1_components)):
			if j != stride_index:
				if layer1_components[j] != layer2_components[j]:
					same = False
		return same

	# Goes through each model's layers and returns total memory of workload
	# and possible memory saved if all merging was successful 
	def total_possible_memory(self):
		start = time.time()
		total_memory = 0.0
		for i in range(len(self.models)):
			model = self.models[i]
			model_mem_usage = summary(model)
			for name, layer in model.named_modules():
				# only consider layers with weights for sharing
				if hasattr(layer, 'weight') and layer.parameters():
					layer_str = str(layer)
					if not (('Conv' in layer_str) or ('Linear' in layer_str)):
						continue

					if layer_str not in model_mem_usage:
						continue
					memory = model_mem_usage[layer_str]
					total_memory += memory 
		possible_memory = 0.0
		for j in self.layer_list:
			mem_for_layer = j['memory']
			total_mem_saved = mem_for_layer * (len(j['indexes']) - 1)
			possible_memory += total_mem_saved
		end = time.time()
		# print(f'Time to get total and possible: {end-start}')
		return total_memory, possible_memory

	# Return whether layer exists already based on our definition of equivalence
	def exists_in_dict(self, layer_dict, layer_str):
		exists = False
		equiv = None
		for i in layer_dict.keys():
			if self.layers_equal(i, layer_str):
				exists = True
				equiv = i
		return exists, equiv


	def layer_instances_from_models(self, models):
		""" Takes a list of models and iterates through them, recording each occurance of a layer in a 
		dict (formatted as indexes, locations, and memory usage, where an occurance cannot be within the same model).
		Then iterates through dict to get a list of the layers that can be shared (appear in more than one location)
		   
		   :param models: list of models
		"""
		layer_dict = {}
		for i in range(len(models)):
			model = models[i]
			model_mem_usage = summary(model)
			for name, layer in model.named_modules():
				# only consider layers with weights for sharing
				if hasattr(layer, 'weight') and layer.parameters():
					layer_str = str(layer)
					if not (('Conv' in layer_str) or ('Linear' in layer_str)):
						continue

					if not layer_str in layer_dict:
						if not layer_str in model_mem_usage:
							continue
						layer_dict[layer_str] = []
						layer_first_instance = {}
						layer_first_instance['indexes'] = [i]
						layer_first_instance['locations'] = [name]
						mem_layer = model_mem_usage[layer_str]
						layer_first_instance['memory'] = mem_layer
						layer_first_instance['layer'] = layer_str

						layer_dict[layer_str].append(layer_first_instance)
					else:
						added = False
						# _, equiv = self.exists_in_dict(layer_dict, layer_str)
						for layer_instance in layer_dict[layer_str]:
							if i not in layer_instance['indexes']:
								layer_instance['indexes'].append(i)
								layer_instance['locations'].append(name)
								added = True
								break
						if not added:
							layer_first_instance = {}
							layer_first_instance['indexes'] = [i]
							layer_first_instance['locations'] = [name]
							layer_first_instance['memory'] = model_mem_usage[layer_str]
							layer_first_instance['layer'] = layer_str

							layer_dict[layer_str].append(layer_first_instance)
		num_instances = 0		
		instance_list = []
		for layer in layer_dict.keys():
			instances = layer_dict[layer]
			for i in instances:
				if len(i['indexes']) > 1:
					i['instance_num'] = num_instances
					instance_list.append(i)
					num_instances += 1
		return instance_list

	# Edit list of merged layers based on successful merges
	def filter_layers_by_indexes(self, layers, indexes):

		new_layer_list = []
		successes_exist = False

		for layer in layers:
			new_layer_dict = {}
			new_layer_dict['indexes'] = [i for i in layer['indexes'] if i in indexes]
			location_indexes = [layer['indexes'].index(item) for item in new_layer_dict['indexes']]
			if new_layer_dict['indexes']:
				successes_exist = True
			new_layer_dict['locations'] = [layer['locations'][i] for i in location_indexes]
			new_layer_dict['memory'] = layer['memory']
			new_layer_dict['layer'] = layer['layer']
			new_layer_dict['instance_num'] = layer['instance_num']
			new_layer_list.append(new_layer_dict)

		if successes_exist:
			return new_layer_list
		else:
			return []

	# Edit list of merged layers based on failed merges
	def filter_out_layers_by_indexes(self, layers, indexes):
		new_layer_list = []
		failures_exist = False
		for layer in layers:
			new_layer_dict = {}
			num_models = len(layer['indexes'])
			new_layer_dict['indexes'] = [i for i in layer['indexes'] if i not in indexes]
			location_indexes = [layer['indexes'].index(item) for item in new_layer_dict['indexes']]
			if new_layer_dict['indexes']:
				failures_exist = True
			else:
				continue
			new_layer_dict['locations'] = [layer['locations'][i] for i in location_indexes]
			new_layer_dict['memory'] = layer['memory']
			new_layer_dict['layer'] = layer['layer']
			new_layer_dict['instance_num'] = layer['instance_num']
			new_layer_list.append(new_layer_dict)
		if failures_exist:
			return new_layer_list
		else:
			return []

	def merge_lists(self, confirmed, current_trial):
		current = confirmed.copy()
		for i, trial_item in enumerate(current_trial):
			instance_num = trial_item['instance_num']
			added = False
			for j, confirmed_item in enumerate(current):
				if instance_num == confirmed_item['instance_num']:
					# Add current to existing
					merged_dict = {}
					new_index_list = confirmed_item['indexes'].copy()
					new_index_list.extend(trial_item['indexes'])
					merged_dict['indexes'] = new_index_list
					merged_indexes = merged_dict['indexes']
					new_loc_list = confirmed_item['locations'].copy()
					new_loc_list.extend(trial_item['locations'])
					merged_dict['locations'] = new_loc_list
					merged_dict['memory'] = confirmed_item['memory']
					merged_dict['layer'] = confirmed_item['layer']
					merged_dict['instance_num'] = current[j]['instance_num']					
					current[j] = merged_dict
					added = True
			if not added:
				current.append(trial_item)
		return current


	def memory_for_config(self, config):
		total_memory = 0
		for layer in config:
			num_times_shared = len(layer['indexes'])
			memory_per_layer = layer['memory']
			total_memory += (memory_per_layer * (num_times_shared - 1))
		return total_memory


	# Returns indexes of models with shared layers
	def indexes_from_config(self, current_config, all_configs):
		all_indexes_current = []
		all_indexes_dependent = []
		for layer in current_config:
			for index in layer['indexes']:
				if index not in all_indexes_current:
					all_indexes_current.append(index)
		for i in all_indexes_current:
			for l in all_configs:
				if i in l['indexes']:
					all_indexes_dependent.extend(l['indexes'])
		return list(set(sorted(all_indexes_dependent)))



	def generate_configs(self):
		pass


	def layers_to_groups(self):
		pass

	# Specifies how to decide which layers will be shared next
	# Returns current config (which layers are shared), potential memory
	# saved by current config, and which of the layers in current config
	# are new to this round of training
	def next_config(self):
		pass


class All_Algorithm(Sharing_Algorithm):
	""" Share all layers
	"""
	def generate_configs(self):
		return self.layer_list

	def layers_to_groups(self, layer_list):
		return

	def next_config(self, indexes_that_met):
		if self.config_index == None:
			_, savings = self.total_possible_memory()
			print(f'savings: {savings}')
			return self.configs, savings, []
		else:
			_, savings = self.total_possible_memory()
			print(f'savings: {savings}')
			return None, savings, None


class Choosing_Algorithm(Sharing_Algorithm):
	""" For testing - manually choose which layers to share in generate_configs
	"""
	def layers_to_groups(self, layer_list):
		return None


	# There is a single config which is just all shareable layers
	def generate_configs(self):
		return self.layer_list


	def memory_for_config(self, config):
		total_memory = 0
		for layer in config:
			num_times_shared = len(layer['indexes'])
			memory_per_layer = layer['memory']
			total_memory += (memory_per_layer * (num_times_shared - 1))
		return total_memory


	# Get the next config to try. There should only be one here
	def next_config(self, indexes_that_met):
		if self.config_index == None:
			current_config = self.configs[:self.num_layers_to_share]
			print(current_config)
		else:
			return None, None, None
		return current_config, self.memory_for_config(current_config), current_config


class Incremental_Greedy_Algorithm(Sharing_Algorithm):
	""" Incrementaly add shareable layers, starting at those with the highest memory
	When a layer is part of a group, add the Group. This one is Gemel's heuristic
	"""
	def __init__(self, shared_layer_manager, models, path_for_storage, k=1):
		self.k = k
		self.confirmed_layers = []
		self.current_trial = []
		super().__init__(shared_layer_manager, models, path_for_storage)

	def layers_to_groups(self, layer_list):
		return None

	def generate_configs(self):
		sorted_layer_list = sorted(self.layer_list, key=lambda x: x['memory'], reverse=True)

		layer_index = 0
		curr_layer_type = sorted_layer_list[0]['layer']
		curr_group = []
		all_groups = []
		while layer_index < len(sorted_layer_list):
			while self.layers_equal(sorted_layer_list[layer_index]['layer'], curr_layer_type):
				curr_group.append(sorted_layer_list[layer_index])
				layer_index += 1
				if layer_index >= len(sorted_layer_list):
					break
			if layer_index < len(sorted_layer_list):
				indexed_dict = {}
				for same_layer in curr_group:
					index_id = '_'.join([str(ind) for ind in same_layer['indexes']])
					if index_id in indexed_dict.keys():
						indexed_dict[index_id].append(same_layer)
					else:
						indexed_dict[index_id] = [same_layer]
				for same_indexes_group in indexed_dict.keys():
					same_indexes_layers = indexed_dict[same_indexes_group]
					mem_for_group = 0.0
					for lay in same_indexes_layers:
						mem = lay['memory'] * (len(lay['indexes']) - 1)
						mem_for_group += mem
					all_groups.append({'group': same_indexes_layers, 'total_memory': mem_for_group})

				curr_layer_type = sorted_layer_list[layer_index]['layer']
				curr_group = []

		sorted_groups = sorted(all_groups, key=lambda x:x['total_memory'], reverse=True)
		for k in sorted_groups:
			tot_mem = k['total_memory']
			print(f'{tot_mem}')
		layers_sorted_by_group_mem = []
		for i in sorted_groups:
			for j in i['group']:
				layers_sorted_by_group_mem.append(j)

		instance_nums = 0
		for i in layers_sorted_by_group_mem:
			i['instance_num'] = instance_nums
			instance_nums += 1
		return layers_sorted_by_group_mem


	def layers_to_groups(self):
		groups = {}
		for index, layer_info in enumerate(self.configs):
			layer = layer_info['layer']
			memory = layer_info['memory']
			layer_exists = False
			for g in groups.keys():
				if self.layers_equal(layer, g):
					layer_exists = True
			if not layer_exists:
				group = self.next_group(index)
				group_memory = group[0]['memory']*(len(group))
				groups[layer] = {'group': group, 'memory':group_memory}
		groups_list = list(groups.values())
		reverse_memory_groups = sorted(groups_list, key=lambda x: x['memory'], reverse=True)
		return reverse_memory_groups
			

	def next_group(self, config_index):
		layer = self.configs[config_index]['layer']
		layer_indexes = self.configs[config_index]['indexes']
		layer_group = []

		i = config_index
		
		while (self.layers_equal(self.configs[i]['layer'], layer)) and (self.configs[i]['indexes'] == layer_indexes):
			layer_group.append(self.configs[i])
			i += 1
			if i >= len(self.configs):
				break

		return i, layer_group


	def next_config(self, indexes_that_met_from_training_pool):
		if self.config_index == None:

			self.config_index, self.current_trial = self.next_group(config_index=0)
			print(f'Starting with {len(self.current_trial)} layers, current config index is {self.config_index}')
		else:
			indexes_that_met = [self.indexes_in_current_config[i] for i in indexes_that_met_from_training_pool]
			print(f'indexes that met: {indexes_that_met}')
			# Get the layers we just tried, and for all that succeeded, add them to the confirmed list
			if len(indexes_that_met) > 1:
				successes_in_current_trial = self.filter_layers_by_indexes(self.current_trial, indexes_that_met)
				if successes_in_current_trial:
					self.confirmed_layers.extend(successes_in_current_trial)
			
			# For the ones that didn't, half the number of layers
			failures = self.filter_out_layers_by_indexes(self.current_trial, indexes_that_met)
			trial_included_failures = 0
			for c in self.current_trial:
				for f in failures:
					if set(f['locations']).issubset(set(c['locations'])):
						if set(f['indexes']).issubset(set(c['indexes'])):
							if f['instance_num'] == c['instance_num']:
								trial_included_failures += 1
			num_layers_to_keep = int(trial_included_failures/2)
			print(num_layers_to_keep)
			
			move_on = False
			_, potential_next_group = self.next_group(self.config_index)
			mem_to_keep_trying = sum([f['memory'] for f in failures[:num_layers_to_keep]])
			mem_to_move_on = self.memory_for_config(potential_next_group)
			if mem_to_keep_trying < mem_to_move_on:
				move_on = True
				print('Moving to next group')

			if (not move_on) and failures and num_layers_to_keep > 0:
				print('Halving number of layers')
				prev_len_current_trial = len(self.current_trial)
				self.current_trial = failures[:num_layers_to_keep]
				self.config_index -= (prev_len_current_trial - num_layers_to_keep)
			
			else:
				# Add next group of layers
				self.config_index, self.current_trial = self.next_group(config_index=self.config_index)
				print(f'Adding {len(self.current_trial)} layers, current config index is {self.config_index}')

					
		self.current_config = self.merge_lists(self.confirmed_layers, self.current_trial)
		self.indexes_in_current_config = self.indexes_from_config(self.current_trial, self.current_config)
		memory = self.memory_for_config(self.current_config)
		return self.current_config, memory, self.current_trial
