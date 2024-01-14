import torch
import sys
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN, MaskRCNN, ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.rpn import AnchorGenerator
# sys.path.append("../dataset_formation/PyTorch-YOLOv3")
# import yolo_model

# All model architectures in the dataset (not trained, need to load weights separately to use)

# Classifiers

def resnet50(num_classes):
    resnet = torchvision.models.resnet50(pretrained=True)
    number_features = resnet.fc.in_features
    resnet.fc = nn.Linear(number_features, num_classes)
    return resnet

def resnet101(num_classes):
    resnet = torchvision.models.resnet101(pretrained=True)
    number_features = resnet.fc.in_features
    resnet.fc = nn.Linear(number_features, num_classes)
    return resnet


def resnet152(num_classes):
    resnet = torchvision.models.resnet152(pretrained=True)
    number_features = resnet.fc.in_features
    resnet.fc = nn.Linear(number_features, num_classes)
    return resnet


def resnet18(num_classes):
	resnet = torchvision.models.resnet18(pretrained=True)
	number_features = resnet.fc.in_features
	resnet.fc = nn.Linear(number_features, num_classes)
	return resnet


def vgg16(num_classes):
    vgg16 = torchvision.models.vgg16(pretrained=True)
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, num_classes)])
    vgg16.classifier = nn.Sequential(*features)
    
    return vgg16


def inceptionv3(num_classes):
	inception = torchvision.models.inception_v3(pretrained=True)
	number_features = inception.fc.in_features
	inception.fc = nn.Linear(number_features, num_classes)
	inception.aux_logits = False

	return inception

# inceptionv3(100)
def mobilenetv3(num_classes):
	mobilenet = torchvision.models.mobilenet_v3_large(pretrained=True)
	num_features = mobilenet.classifier[3].in_features
	mobilenet.classifier[3] = torch.nn.Linear(in_features=num_features, out_features=num_classes)
	return mobilenet

# Object Detectors

# Faster RCNN with Resnet backbone
def resnet18_backbone(): 
	net = torchvision.models.resnet18(pretrained=True)
	modules = list(net.children())[:-2]
	backbone = nn.Sequential(*modules)
	backbone.out_channels = 512
	return backbone


def resnet50_backbone(): 
	net = torchvision.models.resnet50(pretrained=True)
	modules = list(net.children())[:-2]
	backbone = nn.Sequential(*modules)
	backbone.out_channels = 2048
	return backbone


def resnet101_backbone(): 
	net = torchvision.models.resnet101(pretrained=True)
	modules = list(net.children())[:-2]
	backbone = nn.Sequential(*modules)
	backbone.out_channels = 2048
	return backbone


def vgg_backbone():
	backbone = torchvision.models.vgg16(pretrained=True).features
	backbone.out_channels = 512
	return backbone


def mobilenet_backbone():
	backbone = torchvision.models.mobilenet_v2(pretrained=True).features
	backbone.out_channels = 1280
	return backbone


def frcnn_model(backbone):
	anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
								   aspect_ratios=((0.5, 1.0, 2.0),))

	roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
												output_size=7,
												sampling_ratio=2)

	return FasterRCNN(backbone,
				   num_classes=2,
				   rpn_anchor_generator=anchor_generator,
				   box_roi_pool=roi_pooler)


def mask_rcnn_model(backbone):
	anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
								   aspect_ratios=((0.5, 1.0, 2.0),))

	roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
												output_size=7,
												sampling_ratio=2)

	mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=14,
                                                         sampling_ratio=2)

	return MaskRCNN(backbone,
				   num_classes=2,
				   rpn_anchor_generator=anchor_generator,
				   box_roi_pool=roi_pooler,
				   mask_roi_pool=mask_roi_pooler)

# SSD
def ssd_vgg_model():
	return ssd300_vgg16(pretrained_backbone=True, num_classes=2, trainable_backbone_layers=5)

def ssd_mobilenet_model():
	return ssdlite320_mobilenet_v3_large(pretrained_backbone=True, num_classes=2, trainable_backbone_layers=6)

# YOLOv3
def yolov3():
	return yolo_model.Darknet(config_path="../dataset_formation/PyTorch-YOLOv3/config/yolov3-custom.cfg")


def tiny_yolov3():
	return yolo_model.Darknet(config_path="../dataset_formation/PyTorch-YOLOv3/config/yolov3-tiny-custom.cfg")

