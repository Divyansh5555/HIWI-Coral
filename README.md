# Deep Learning Models for Coral and Polyp Segmentation

This branch, `deeplearning_models`, provides the complete training and testing pipelines for three deep learning-based semantic segmentation models, designed for application on **Coral** and **Polyp** image datasets:

- U-Net  
- U-Net++  
- DeepLabV3++  

The objective is to evaluate and compare these models on biomedical segmentation tasks, with emphasis on segmentation accuracy, generalization capability, and inference performance.

---

## Branch Overview

### `deeplearning_models`

This branch includes all core components for model development and evaluation:

- Modular implementations of U-Net, U-Net++, and DeepLabV3++
- Scripts for training and testing
- Dataset support for both Coral and Polyp image datasets
- Metric tracking: Dice score, Intersection over Union (IoU), loss, and average inference time
- Visualization utilities for comparing predicted masks with ground truth annotations

### `frontend_dash`

This branch contains the source code for the web-based semi-automated segmentation tool developed using **Dash**. The tool provides a graphical user interface for:

- Uploading and displaying input images
- Viewing model predictions and overlays
- Interactively refining segmentation outputs
- Generating basic quantification results (e.g., area, pixel counts)

This interface facilitates easier interaction with the trained models, making the tool suitable for usability testing and potential deployment in applied research contexts.

---

## Models

### `unet.py`

Implements the original U-Net architecture, featuring an encoder–decoder structure with skip connections. Designed for biomedical segmentation tasks and known for strong performance with limited data.

### `unetpp.py`

Provides the U-Net++ architecture, which introduces nested and dense skip pathways to enhance feature propagation and improve boundary delineation in segmentation tasks.

### `deeplabv3pp.py`

Implements the DeepLabV3++ architecture, optionally utilizing EfficientNetV2-S as the encoder. This model supports additional custom modules (e.g., MSPP and PAAB) for multi-scale feature extraction and improved contextual understanding.
