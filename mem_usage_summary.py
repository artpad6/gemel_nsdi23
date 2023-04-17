import sys
import time
import torch
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

from models.model_architectures import resnet50_backbone, frcnn_model, resnet50, resnet18

def summary_string(model, input_size, batch_size=2, device=torch.device('cpu'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = str(module)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].shape)

            summary[m_key]["input_shape"][0] = batch_size
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params


        if hasattr(module, 'weight') and module.parameters():
            hooks.append(module.register_forward_hook(hook))
        

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]
        

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15} {:>15}".format(
        "Layer (type)", "Output Shape", "Output Size", "Param #", "Total (GB)")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    layer_usage_dict = {}
    for layer in summary:
        layer_usage_dict[layer] = float(abs(summary[layer]["nb_params"] * 4. / (1024 **2.)))
        
        total_params += summary[layer]["nb_params"]
    total_params_size = abs(total_params * 4. / (1024 ** 2.))

    x = [y.to(torch.device('cpu')) for y in x]
    
    return layer_usage_dict


def summary(model, input_size=(3, 224, 224), batch_size=4, device=torch.device('cpu'), dtypes=None):
    
    model = model.to(device)
    model.eval()
    
    usage_dict = summary_string(model, input_size, batch_size, device, dtypes)
    model = model.to('cpu')
    torch.cuda.empty_cache() 

    return usage_dict   