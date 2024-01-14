import os
import sys
import json
import torch
import time
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.hub import load_state_dict_from_url
import torch.nn as nn
from model_merger import ModelMerger
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from collections import OrderedDict
from itertools import combinations

from transforms import FRCNN_TRANSFORM, YOLOv3_TRAIN_TRANSFORM, YOLOv3_VAL_TRANSFORM, classification_transforms
from eval_methods import frcnn_eval, yolo_eval, classification_eval
from models.model_architectures import resnet50_backbone, resnet101_backbone, frcnn_model, resnet50, resnet101, vgg16, resnet152, resnet18, mobilenetv3, inceptionv3, ssd_vgg_model, ssd_mobilenet_model, yolov3, tiny_yolov3

# Expects a dict containing each model to be merged and some info about it (see README and example below)
def merge_workload(model_dict):
	for key in model_dict.keys():
		unmerged_acc = model_dict[key]['unmerged_acc']
		print(f'{key}: {unmerged_acc}')
	
	# Store the merged results in a results folder
	path = f'results'
	if not os.path.exists(path):
		os.mkdir(path)

	merger = ModelMerger(model_dict, path)
	merger.merge()

# An example of a model dict with two tasks (entries)
# One classifies cars vs. people at Main & 2nd with ResNet50
# One classifies cars, trucks, and motorcycles at 1st & Elm with ResNet101
def create_sample_model_dict():
	model_dict = {}
	
	# Task 1
	model_dict['main_2nd_cars_people_resnet50'] = {}
	model_dict['main_2nd_cars_people_resnet50']['unmerged_acc'] = 0.97

	# Initialize model structure and load weights
	model_main_2nd = resnet50(2) # 2 classes
	model_main_2nd.load_state_dict(torch.load('main_2nd_cars_people_resnet50_weights.pt'))
	model_dict['main_2nd_cars_people_resnet50']['model'] = model_main_2nd

	model_dict['main_2nd_cars_people_resnet50']['task'] = 'image_classification'
	model_dict['main_2nd_cars_people_resnet50']['eval_method'] = classification_eval
	model_dict['main_2nd_cars_people_resnet50']['transforms'] = {'train': classification_transforms, 'val': classification_transforms}

	# Task 2
	model_dict['elm_1st_cars_trucks_motorcycles_resnet101'] = {}
	model_dict['elm_1st_cars_trucks_motorcycles_resnet101']['unmerged_acc'] = 0.99

	# Initialize model structure and load weights
	model_elm_1st = resnet101(3) # 3 classes
	model_elm_1st.load_state_dict(torch.load('elm_1st_cars_trucks_motorcycles_resnet101_weights.pt'))
	model_dict['elm_1st_cars_trucks_motorcycles_resnet101']['model'] = model_elm_1st

	model_dict['elm_1st_cars_trucks_motorcycles_resnet101']['task'] = 'image_classification'
	model_dict['elm_1st_cars_trucks_motorcycles_resnet101']['eval_method'] = classification_eval
	model_dict['elm_1st_cars_trucks_motorcycles_resnet101']['transforms'] = {'train': classification_transforms, 'val': classification_transforms}

	return model_dict

def main():
	model_dict = create_sample_model_dict()
    merge_workload(model_dict)

if __name__ == "__main__":
    main()