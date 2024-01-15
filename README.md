To get started merging models, run merge_workloads.py. Below is the setup needed before this will work.

### 1) Create a dictionary of tasks to merge.

Dictionary keys are the task names, e.g., main_2nd_cars_people_resnet50 to classify cars vs. people at Main & 2nd using a ResNet50 model. We recommend the same naming convention, or at least putting an underscore and the type of model at the end (_resnet50), as the Dataset Manager expects the dataset name to have a matching format (see Datasets below) See create_sample_model_dict in merge_workloads for a complete example. Within each job, the inner dictionary must have the following:

***model***: a PyTorch model (.pt) with weights loaded

***task***: task type, i.e., “object detection” or “image classification” (these are the only options currently supported)

***eval_method***: We need to periodically check the each model’s accuracy during training, so the dictionary expects a method that takes a model and a [data loader object](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). We’ve included some common ones in eval_methods.py but you can add your own method for a new model

***transforms***: You must provide the ‘train’ and ‘val’ transforms that need to run on the dataset before it can be run through the model. This dict entry must take a dict containing ‘train’ and ‘val’. See example in merge_workloads

***unmerged_acc***: To assess how well the model is doing during training, we need to know the “ground truth” accuracy, or how well it was performing on the dataset before any merging. This should be in the form of a decimal

### 2) Add your datasets 

In the same folder as gemel_nsdi23 (but not within it), create a dataset_formation folder. Within that, create a datasets folder. Within datasets, the code expects folders named with task name, minus the specific model (because the same tasks using Resnet50 or ResNet101, for example, will use the same dataset) plus “OD” or “CL” for object detection or image classification. For the example above, it is expected that the folder “main_2nd_cars_people_CL” exists in datasets.

### Results ###
After setting this up and running, you’ll see training and validation output, and the results of the merging will be in the ‘results’ folder (you can change the location in merge_workloads). This will include all the models you started with but now the weights of the merged layers are the same. 

TODO: Add functionality to also store which layers were saved in the results folder in a format such that we can then piece the merged models back together using their new (shared) weights

> [!NOTE]
> This repo as is only supports models in the torchvision models library as of early 2022. To add other models, you can add them to model_architectures and add eval_methods and transforms. The models’ inputs and outputs must match torchvision models (see FasterRCNN [here](https://pytorch.org/vision/0.9/models.html#torchvision.models.detection.fasterrcnn_resnet50_fpn).) We did this for YOLOv3 by editing a version of [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) from early 2021. Please email arpadmanabhan@g.hmc.edu for more detail about this YOLOv3 implementation, but this code is not published because we do not maintain it.)
