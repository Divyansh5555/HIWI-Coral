Deep Learning Models for Coral & Polyp Segmentation

This branch, deeplearning_models, includes the full training and testing setup for three deep learning models applied to Coral and Polyp image datasets:
	•	U-Net
	•	U-Net++
	•	DeepLabV3++

The main goal is to evaluate and compare these models on biomedical segmentation tasks, with a focus on accuracy, generalization, and inference time.

⸻

 Branch Overview

deeplearning_models

Contains all model architectures, training scripts, and evaluation tools. This branch is focused on model development, experimentation, and quantitative analysis.

What’s inside:
	•	Modular files for U-Net, U-Net++, and DeepLabV3++
	•	Training and testing scripts
	•	Dataset support for both Coral and Polyp
	•	Metric logging (Dice, IoU, loss, and inference time)
	•	Visual comparison of predicted vs. ground truth masks

frontend_dash

This branch contains the frontend code used to build a semi-automated Dash tool. It allows users to interact with the segmentation models through a web interface—uploading images, viewing predictions, refining results, and getting quick visual feedback. It’s ideal for making the models more accessible and usable in practical settings.

⸻

 Models

1. unet.py

Standard encoder–decoder architecture for image segmentation. Lightweight and effective, especially on biomedical datasets.

2. unetpp.py

Improved U-Net with dense skip connections and nested architecture, offering better segmentation performance on complex boundaries.

3. deeplabv3pp.py

Advanced segmentation model using atrous convolutions and optional EfficientNetV2-S encoder. Supports extensions like MSPP and PAAB for enhanced multi-scale context.

