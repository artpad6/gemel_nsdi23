import os
import sys
import time
import torch
import numpy as np
from torchvision.ops import nms
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from joint_trainer import move_to_gpu

sys.path.append("../dataset_formation/PyTorch-YOLOv3")
from utils.utils import *
from utils.datasets import *

def frcnn_eval(model, dataloader):
	coco = get_coco_api_from_dataset(dataloader.dataset)
	iou_types = ["bbox"]
	coco_evaluator = CocoEvaluator(coco, iou_types)
	
	for inputs, targets in dataloader:
		inputs, targets = move_to_gpu(inputs, targets)
		outputs = model(inputs)

		nms_thresholded_outputs = []
		for output in outputs:
			boxes = output['boxes']
			labels = output['labels']
			scores = output['scores']
			keep_indices = nms(boxes, scores, 0.5)
			keep_boxes_nms = [boxes[i] for i in keep_indices]
			keep_scores_nms = [scores[i] for i in keep_indices]
			keep_labels_nms = [labels[i] for i in keep_indices]
			keep_boxes_tensor = torch.stack(keep_boxes_nms) if keep_boxes_nms else torch.tensor([])
			keep_boxes_tensor = keep_boxes_tensor.reshape(-1, 4)
			nms_thresholded_outputs.append({'boxes': keep_boxes_tensor, 
				'labels': torch.stack(keep_labels_nms) if keep_labels_nms else torch.tensor([]), 
				'scores': torch.stack(keep_scores_nms) if keep_scores_nms else torch.tensor([])})
		nms_thresholded_outputs = [{k: v.to('cpu') for k, v in t.items()} for t in nms_thresholded_outputs]
		res = {target["image_id"].item(): output for target, output in zip(targets, nms_thresholded_outputs)}
		coco_evaluator.update(res)
		
	coco_evaluator.synchronize_between_processes()
	coco_evaluator.accumulate()
	# use logic from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
	# to get the single mAP score
	p = coco_evaluator.coco_eval['bbox'].params
	s = coco_evaluator.coco_eval['bbox'].eval['precision']
	
	aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == 'all']
	mind = [i for i, mDet in enumerate(p.maxDets) if mDet == 100]
	s = s[:,:,:,aind,mind]
	if len(s[s>-1])==0:
		mean_s = -1
	else:
		mean_s = np.mean(s[s>-1])	

	return mean_s


def yolo_eval(model, dataloader):
	img_size = 416
	labels = []
	sample_metrics = []  # List of tuples (TP, confs, pred)
	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
	index = 0
	model = model.to('cuda')
	for imgs, targets in dataloader:
		imgs = torch.stack(imgs)
		imgs = imgs.to('cuda')

		for i, boxes in enumerate(targets):
			boxes[:, 0] = i
		targets = torch.cat(targets, 0)

		if targets is None:
			print('continuing')
			continue
			
		# Extract labels
		labels += targets[:, 1].tolist()
		# Rescale target
		targets[:, 2:] = xywh2xyxy(targets[:, 2:])
		targets[:, 2:] *= img_size

		imgs = Variable(imgs.type(Tensor), requires_grad=False)

		with torch.no_grad():
			outputs = model(imgs)
			outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.5)


		sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)


		index += 1
	
	if len(sample_metrics) == 0:  # no detections over whole validation set.
		print('no sample metrics')
		return None
	
	# Concatenate sample statistics
	true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
	precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

	return AP.mean()


def classification_eval(model, dataloader):
	running_corrects = 0
	total = 0
	for inputs, targets in dataloader:
		inputs, targets = move_to_gpu(inputs, targets)
		inputs = torch.stack(inputs)
		targets = torch.FloatTensor(targets).to('cuda')

		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		running_corrects += torch.sum(preds == targets.data)
		total += len(targets)
	classifier_acc = running_corrects.double()/total
	return classifier_acc.item()



