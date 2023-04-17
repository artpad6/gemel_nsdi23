import os
import sys
import torch
import torchvision
from torchvision import transforms

sys.path.append("../dataset_formation/PyTorch-YOLOv3")
from utils.augmentations import *
from utils.transforms import *


class FRCNN_ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, detections, image_id = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)


        boxes = [detection[1:] for detection in detections]
        num_detections = len(boxes)


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.reshape(-1, 4)
        labels = torch.ones((num_detections,), dtype=torch.int64)
        areas = []
        for i in range(num_detections):
            areas.append((boxes[i][2] - boxes[i][0])*(boxes[i][3] - boxes[i][1]))
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_detections,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        return img, target


FRCNN_TRANSFORM = transforms.Compose([FRCNN_ToTensor()])

class YOLOv3_Train_Transforms_List(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes, image_id = data
        img, boxes = AUGMENTATION_TRANSFORMS((np.array(img), boxes))
        resize_transform = Resize(416)
        img, boxes = resize_transform((img, boxes))


        return img, boxes


YOLOv3_TRAIN_TRANSFORM = transforms.Compose([YOLOv3_Train_Transforms_List()])


class YOLOv3_Val_Transforms_List(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes, image_id = data
        img, boxes = DEFAULT_TRANSFORMS((np.array(img), boxes))
        resize_transform = Resize(416)
        img, boxes = resize_transform((img, boxes))
        return img, boxes


YOLOv3_VAL_TRANSFORM = transforms.Compose([YOLOv3_Val_Transforms_List()])

def classification_transforms(img):
    transforms_list = []
    transforms_list.append(torchvision.transforms.Resize(256))
    transforms_list.append(torchvision.transforms.CenterCrop(224))
    transforms_list.append(torchvision.transforms.ToTensor())
    transforms_list.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transforms_compose = torchvision.transforms.Compose(transforms_list)
    return transforms_compose(img)

def inception_transforms(img):
    transforms_list = []
    transforms_list.append(torchvision.transforms.Resize(299))
    transforms_list.append(torchvision.transforms.CenterCrop(299))
    transforms_list.append(torchvision.transforms.ToTensor())
    transforms_list.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transforms_compose = torchvision.transforms.Compose(transforms_list)
    return transforms_compose(img)

