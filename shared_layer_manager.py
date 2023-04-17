import os
import time
import torch
import copy
from sharing_algorithm import Incremental_Greedy_Algorithm


class SharedLayerManager():
    """ Handles finding layers that can be shared, merging architecture, and managing copies of layers
    
    :param models: list of models to find shared architecture
    """
    def __init__(self, models, model_names, path_for_storage):
        self.models = models
        self.replaced_layers = []
        self.model_names = model_names
        self.alg = Incremental_Greedy_Algorithm(self, models, path_for_storage)
       
    def refresh_models(self, results_path, replace_weights=True):
        for layer in self.replaced_layers:
            model = self.models[layer[0]]
            module_name = layer[1]
            

            module_components = module_name.split('.')
            module = model
            for i in range(len(module_components)-1):
                comp = module_components[i]
                module = module._modules[comp]
            layer_copy = copy.deepcopy(module._modules[module_components[-1]])
            module._modules[module_components[-1]] = layer_copy

        if replace_weights:
            for i in range(len(self.models)):
                model_name = self.model_names[i]
                self.models[i].load_state_dict(torch.load(os.path.join(results_path, 'weights', f'{self.model_names[i]}.pt')))

    # Retrieves PyTorch module (layer) from layer name (e.g., conv1)  
    def layer_from_name(self, model, module_name):
        module_components = module_name.split('.')
        module = model
        for comp in module_components:
            module = module._modules[comp]
        return module

    # Take all shared layers and make all models share a single copy
    def share_layers(self, models, shared_layers):
        replaced_layers = []
        for sharing_instance in shared_layers:
            # one layer to be shared in multiple locations
            replaced_with_index = sharing_instance['indexes'][0]
            replaced_with_loc = sharing_instance['locations'][0]
            num_locations = len(sharing_instance['indexes'])
            for i in range(1, num_locations):
                replacing_index = sharing_instance['indexes'][i]
                replacing_loc = sharing_instance['locations'][i]
                models[replacing_index] = self.replace_layer(models[replacing_index], replacing_index, replacing_loc, models[replaced_with_index], replaced_with_index, replaced_with_loc)
                replaced_layers.append((replacing_index, replacing_loc))
        return models, replaced_layers

    # Replace a layer in a model with the layer from another model
    def replace_layer(self, model, model_index, module_name, model_replacement, replacement_index, module_name_replacement):
        module_replacement = self.layer_from_name(model_replacement, module_name_replacement)
        print(f'Replacing {module_name} from model {model_index} with {module_name_replacement} from model {replacement_index}')
        module_components = module_name.split('.')
        module = model
        for i in range(len(module_components)-1):
            comp = module_components[i]
            module = module._modules[comp]
        module._modules[module_components[-1]] = module_replacement
        return model


    # Returns the indexes of models with any shared layers because only
    # those need to be trained
    def indexes_shared_models(self, current_config, full_config):
        all_indexes_current = []
        all_indexes_dependent = []
        for layer in current_config:
            for index in layer['indexes']:
                if index not in all_indexes_current:
                    all_indexes_current.append(index)
        for i in all_indexes_current:
            for l in full_config:
                if i in l['indexes']:
                    all_indexes_dependent.extend(l['indexes'])
        return list(set(sorted(all_indexes_dependent)))


    def total_possible_memory(self):
        return self.alg.total_possible_memory()

    # Updates merged layers for the next round of training based on which layers successfully merged 
    def update_merging(self, indexes_that_met):
        updated_shared_layers, memory_savings, current_trial = self.alg.next_config(indexes_that_met)
        if updated_shared_layers == None:
            return None, None

        indexes_for_training = self.indexes_shared_models(current_trial, updated_shared_layers)
        models, replaced_layers = self.share_layers(self.models, updated_shared_layers)

        self.replaced_layers = replaced_layers

        # Separate out just layers
        layer_list = []
        for i in updated_shared_layers:
            if i not in current_trial:
                layer_list.append((i['layer'], i['indexes'], i['locations']))

        return models, indexes_for_training, memory_savings, layer_list

        

        