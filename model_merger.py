import os
import time
import json
import torch
from dataset_manager import DatasetManager
from shared_layer_manager import SharedLayerManager
from joint_trainer import JointTrainer
from torch.utils.data import DataLoader, ConcatDataset


class ModelMerger():
    """ Handles end-to-end process of merging models
            
        Args:
          model_dict: dictionary of models and related info (see README for dict format)                
    """

    def __init__(self, model_dict, results_path):  
        # Checks model dict format
        for entry in model_dict.keys():
            model_info = model_dict[entry]
            assert('model' in model_info)
            assert('task' in model_info)
            assert('transforms' in model_info)
            transforms = model_info['transforms']
            assert('train' in transforms)
            assert('val' in transforms)
            assert('unmerged_acc' in model_info)
            
        self.model_dict = model_dict
        self.results_path = results_path
        self.dataset_manager = DatasetManager(self.model_dict)
        self.shared_layer_manager = SharedLayerManager([model_dict[entry]['model'] 
            for entry in model_dict.keys()], [entry for entry in model_dict.keys()], results_path)
        self.joint_trainer = JointTrainer()

        self.save_model_weights(results_path)
     

    def save_model_weights(self, results_path):
        for entry in self.model_dict.keys():
            model = self.model_dict[entry]['model']
            weights_path = os.path.join(results_path, 'weights')
            if not os.path.exists(weights_path):
                os.mkdir(weights_path)
            torch.save(model.state_dict(), os.path.join(weights_path,f'{entry}.pt'))


    def refresh_models(self):
        self.shared_layer_manager.refresh_models(self.results_path)

    def calculate_mem_savings(self, potential_mem_savings, num_models, prev_savings, num_models_met):
        # calculate total possible from this round
        possible_mem_this_round = potential_mem_savings - prev_savings
        # calculate indexes over num model
        actual_mem_savings = (possible_mem_this_round/(num_models-1))*(num_models_met-1)

        return actual_mem_savings

    def add_mem_savings(self, curr_time, indexes_over_time_round, mem_savings_over_time, potential_mem_savings, num_models, prev_potential_savings, prev_savings):
        for pair in indexes_over_time_round:
            num_models_met = len(pair[0])
            time_within_round = pair[1]

            mem_savings = prev_savings + self.calculate_mem_savings(potential_mem_savings, num_models, prev_savings, num_models_met)
            time = curr_time + time_within_round
            mem_savings_over_time.append((mem_savings, time))

        return mem_savings_over_time

    def merge(self, lr=0.0001):
        """Joins all parts of the merging process

        Creates a joint dataset, merges shared layers, jointly trains models and
        returns list of all model val accuracies at the end of each epoch

        Args:
          lr: learning rate for training
        """
        num_entries = len(self.model_dict.keys())
        tasks = [self.model_dict[entry]['task'] for entry in self.model_dict.keys()]
        unmerged_accs = [self.model_dict[entry]['unmerged_acc'] for entry in self.model_dict.keys()]
        eval_methods = [self.model_dict[entry]['eval_method'] for entry in self.model_dict.keys()]
        dataset_names = list(self.model_dict.keys())

        indexes_that_met = None
        # Keep track of actual and potential savings to assess how much was saved each round
        actual_memory_savings = 0.0
        potential_mem_savings = 0.0
        start = time.time()
        mem_savings_over_time = []

        total_mem, possible_mem = self.shared_layer_manager.total_possible_memory()
        print(f'Total memory before merging: {total_mem}, possible savings: {possible_mem}')

        while True:
            prev_savings = actual_memory_savings
            prev_potential_savings = potential_mem_savings
            self.refresh_models()
            models, indexes_for_training, potential_mem_savings, layers_in_config = self.shared_layer_manager.update_merging(indexes_that_met)
            print(f'Right after refresh, indexes for training: {indexes_for_training}')

            # Save stats from last run based on on which layers we successfully found weights for
            # layers in config includes only layers that are already successful, not the potential layer from the upcoming round
            with open(os.path.join(self.results_path, 'workload_stats.json'), 'w') as stats_file:
                stats_dict = {'total_mem': total_mem, 'possible_mem': possible_mem, 'mem_time': mem_savings_over_time, 'layers': layers_in_config}
                json.dump(stats_dict, stats_file)

            # Concatenates all datasets and gets dataloaders for it
            train_dataloader, val_dataloaders = self.dataset_manager.dataloaders(indexes_for_training)
            training_models = [models[i] for i in indexes_for_training]
            training_tasks = [tasks[i] for i in indexes_for_training]
            training_unmerged_accs = [unmerged_accs[i] for i in indexes_for_training]
            training_eval_methods = [eval_methods[i] for i in indexes_for_training]
            training_dataset_names = [dataset_names[i] for i in indexes_for_training]
            print(f'Training on these datasets: {training_dataset_names}')
            

            print(f'Running config with potential memory savings {potential_mem_savings} ({potential_mem_savings - prev_savings} from this round)')
            time_before_training = time.time() - start
            
            indexes_over_time = self.joint_trainer.train(models=training_models, train_dataloader=train_dataloader, 
            val_dataloaders=val_dataloaders, tasks=training_tasks, unmerged_accs=training_unmerged_accs, eval_methods=training_eval_methods, dataset_names=training_dataset_names, results_path=self.results_path)
            
            mem_savings_over_time = self.add_mem_savings(time_before_training, indexes_over_time, mem_savings_over_time, potential_mem_savings, len(indexes_for_training), prev_potential_savings, prev_savings)
            if mem_savings_over_time:
                actual_memory_savings = mem_savings_over_time[-1][0]
            else:
                actual_memory_savings = 0.0            
            if indexes_over_time:
                indexes_that_met = indexes_over_time[-1][0]
            else:
                indexes_that_met = []
            print(mem_savings_over_time)


            
        
        
