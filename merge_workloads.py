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
from eval_methods import frcnn_eval, yolo_eval
from models.model_architectures import resnet50_backbone, resnet101_backbone, frcnn_model, resnet50, resnet101, vgg16, resnet152, resnet18, mobilenetv3, inceptionv3, ssd_vgg_model, ssd_mobilenet_model, yolov3, tiny_yolov3
from form_workloads import dict_for_workload

def merge_workload(wl_num):
	model_dict = dict_for_workload(wl_num, continue_from_checkpoint=False)
	for key in model_dict.keys():
		unmerged_acc = model_dict[key]['unmerged_acc']
		print(f'{key}: {unmerged_acc}')
	path = f'results_generalizations/object/frcnn-r50'
	if not os.path.exists(path):
		os.mkdir(path)

	merger = ModelMerger(model_dict, path)
	merger.merge()

for i in range(0, 1):
	merge_workload(i)