#  Deep Learning Models for Coral & Polyp Segmentation

This branch `deeplearning_models` contains the complete training and testing pipelines for three deep learning segmentation models applied to **Coral** and **Polyp** image datasets:

- U-Net
- U-Net++
- DeepLabV3++

The goal is to evaluate and compare these models for biomedical segmentation tasks, focusing on accuracy, generalization, and inference efficiency.

---

##  Branch Overview: `deeplearning_models`

This branch includes:
- Modular model files for U-Net, U-Net++, and DeepLabV3++
- Scripts for training and evaluation
- Support for both Coral and Polyp datasets
- Metric logging (Dice, IoU, Loss, Time)
- Visualization of predictions alongside ground truth

---

##  Models

### 1. `UNet.py`
Baseline encoder-decoder architecture for segmentation.

### 2. `UNet++.py`
U-Net++ with redesigned skip pathways and nested dense blocks.

### 3. `Deeplabv3++.py`
DeepLabV3++ with optional EfficientNetV2-S backbone and custom decoder (MSPP + PAAB, optional).

