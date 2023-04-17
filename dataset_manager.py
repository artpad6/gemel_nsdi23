import cv2
import json
import os
import bisect
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.utils import save_image
from torchvision.ops import nms


class DatasetManager():
    """Forms datasets used for merging
    
    :param model_dict: dict with model info; parts needed here: task, transforms, phase (train or val)
    """
    def __init__(self, model_dict):
        self.model_dict = model_dict


    # since model_dict is an OrderedDict, we know that datasets will be in right order
    def datasets(self):
        dataset_list = []
        training_phases = ['train', 'val']
        for entry in self.model_dict.keys():
            model_entry = self.model_dict[entry]
            task = model_entry['task']
            dataset_dict = {}
            if task == 'object_detection':
                for phase in training_phases:
                    dataset_dict[phase] = ObjectDetectionDataset(model_name=entry, 
                        model_info=model_entry, training_phase=phase)
            elif task == 'image_classification':
                for phase in training_phases:
                    dataset_dict[phase] = ImageClassificationDataset(model_name=entry,
                        model_info=model_entry, training_phase=phase)
            dataset_list.append(dataset_dict)
        return dataset_list

    
    # return index along with image and target
    def train_collate(self, batch):
        img = [item[0][0] for item in batch]
        target = [item[0][1] for item in batch]
        indexes = [item[1] for item in batch]
        return (img, target), indexes
    
    # same as default collate except that targets might be different lengths so don't stack them
    def val_collate(self, batch):
        # img = torch.stack([item[0] for item in batch])
        img = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return img, target
    
   
    def dataloaders(self, model_indexes, train_batch_size=2, val_batch_size=1):
        """Create dataloaders for joint training and validation. Train dataloader is a single loader that 
        includes all images and targets for joint training (randomly sampling each batch). Validate separately 
        # though, so val_loaders is a list of dataloaders, one per dataset
        
        """
        datasets_all = self.datasets()

        selected_datasets = [datasets_all[i] for i in model_indexes]
        
        
        train_loader = DataLoader(ConcatWithIndex([dataset['train'] for dataset in selected_datasets]), batch_size=train_batch_size,
                                                 shuffle=True, num_workers=0, collate_fn=self.train_collate)
        val_loaders = [DataLoader(dataset['val'], batch_size=val_batch_size, shuffle=True, num_workers=0, collate_fn=self.val_collate)
                      for dataset in selected_datasets]
        
        return train_loader, val_loaders
        
        
class ConcatWithIndex(ConcatDataset):
    """Concatenate all dataset and return image/target and dataset index so we can train differently for each dataset
    
    """
    def __init__(self, datasets):
        super(ConcatWithIndex, self).__init__(datasets)
        
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx   
    
    
class ObjectDetectionDataset():
    """Creates dataset for object detection
    
    """
    def __init__(self, model_name, model_info, training_phase):
        dataset_parts = model_name.split('_')
        dataset_parts[-1] = 'OD'
        dataset_name = '_'.join(dataset_parts)
        dataset_path = os.path.join('..', 'dataset_formation', 'datasets', f'{dataset_name}')

        # The default format for annotations follows YOLO (x-c, y-c, w, h), but some models (frcnn
        # and ssd) require data in a different format (x-min, y-min, x-max, y-max)
        self.need_format_edit = ('frcnn' in model_name or 'ssd' in model_name)
        
        self.transforms = model_info['transforms'][training_phase]
        self.images_path = os.path.join(dataset_path, f'{training_phase}', 'images')
        self.annotations_path = os.path.join(dataset_path, 
                f'{training_phase}', 'annotations')
        self.images = list(sorted(os.listdir(self.images_path)))
        self.annotations = list(sorted(os.listdir(self.annotations_path)))
        self.box_format = model_info['box_format'] if 'box_format' in model_info else 'XMIN_YMIN_XMAX_YMAX'
        self.background_class = model_info['background_class'] if 'background_class' in model_info else True

    # change format from YOLO's cls x-center y-center width height to frcnn cls x-min y-min x-max y-max
    def edit_annotation_format(self, detections, size):
        width, height = size
        edited_detections = []
        for det in detections:
            box_info = [float(i) for i in det.split(' ')]
            xmin = max(0.0, box_info[1] - (box_info[3] / 2.0))
            ymin = max(0.0, box_info[2] - (box_info[4] / 2.0))
            xmax = min(width, xmin + box_info[3])
            ymax = min(height, ymin + box_info[4])
            edited_detections.append([int(box_info[0]), xmin * width, ymin * height, xmax * width, ymax * height])
        return edited_detections

        
    # get image and annotation and convert file contents to target that model expects to train
    def __getitem__(self, index):
        image_path = f'{self.images_path}/{self.images[index]}'
        annotation_path = f'{self.annotations_path}/{self.annotations[index]}'
        
        # Get image from path
        image = Image.open(image_path).convert("RGB") 

        # Get boxes from annotation       
        if self.need_format_edit:
            with open(annotation_path, 'r') as ann_file:
                boxes = ann_file.readlines()
                boxes = self.edit_annotation_format(boxes, image.size)
                boxes = np.asarray([np.asarray(f) for f in boxes])
                boxes = boxes.reshape(-1, 5)
                image, target = self.transforms((image, boxes, index))

        else:
            boxes = np.loadtxt(annotation_path)
            boxes = boxes.reshape(-1, 5)
            image, target = self.transforms((image, boxes, index))

        return image, target
    
    def __len__(self):
        return len(self.images)
    

class ImageClassificationDataset(datasets.ImageFolder):
     """Creates dataset for image classification
    
    """
    def __init__(self, model_name, model_info, training_phase):
        dataset_parts = model_name.split('_')
        dataset_parts[-1] = 'CL'
        dataset_name = '_'.join(dataset_parts)
        dataset_path = os.path.join('..', 'dataset_formation', 'datasets', f'{dataset_name}')
        root_path = os.path.join(dataset_path, training_phase)
        super().__init__(root=root_path, transform=model_info['transforms'][training_phase])

