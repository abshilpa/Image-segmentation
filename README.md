#  Semantic Image Segmentation with U-Net vs DeepLabV3+

This repository contains my image segmentation project where I implemented, trained, and evaluated two popular deep learning models — **U-Net** and **DeepLabV3+** —on a subset of the COCO 2017 dataset. The focus was on multi-class semantic segmentation for classes: `cake`, `car`, `dog`, and `person`.


##  Project Overview

Semantic segmentation is a crucial task in computer vision, aiming to assign a class label to each pixel in an image. In this project, I explored:

* Building **U-Net from scratch** in PyTorch
* Adapting **DeepLabV3+** using `torchvision.models.segmentation`
* Preprocessing COCO data for selected classes
* Evaluating model performance using **IoU**, **Accuracy**, and **Visual Outputs**

##  Dataset

* Source: [COCO 2017](https://cocodataset.org/)
* Filtered 4 classes: `cake`, `car`, `dog`, `person`
* Image Resolution: 256×256 (resized for efficiency)
* Dataset Splits:

  * Train: 300 images
  * Validation: 300 images
  * Test: 30 images (for final visualization only)

##  Model Architectures

###  U-Net (From Scratch)

* Custom encoder-decoder design
* Skip connections for fine-detail preservation
* Trained using `CrossEntropyLoss` and `Adam` optimizer

###  DeepLabV3+ (ResNet50 Backbone)

* Loaded from `torchvision.models.segmentation`
* ASPP module to capture multi-scale context
* Modified final head to classify 5 classes

# Challenges

* Faced **GPU memory limitations** in Colab, restricting training epochs
* Slight variation in results across runs due to runtime interruptions

##  Key Findings

* U-Net: Better for fast, small-data setups; smooth predictions
* DeepLabV3+: Better boundary accuracy and generalization on complex scenes




