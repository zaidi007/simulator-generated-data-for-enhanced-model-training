Simulator-generated-data-for-enhanced-model-training

# Faster R-CNN Object Detection Project

This repository contains a Faster R-CNN implementation for object detection tasks. The project trains a Faster R-CNN model using a custom dataset and evaluates its performance.

## Project Overview
The goal of this project is to detect objects in images by training a Faster R-CNN model. I am using CARLA-generated traffic sign images and real traffic sign images to train and test datasets of real, real+AI, and AI-generated images. The objective is to determine whether adding AI-generated images (CARLA-generated) enhances detection performance.

## Features
- Customizable Faster R-CNN model configuration.
- Support for COCO-format datasets.

## Requirements
The project was developed and tested with the following dependencies:

- Python 3.x
- PyTorch
- torchvision
- COCO API (`pycocotools`)
- Pillow
- NumPy

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses datasets in COCO format. Ensure your dataset is structured as follows:
```
path/to/dataset/
├── train/
│   ├── images/
│   ├── annotations/
├── val/
│   ├── images/
│   ├── annotations/
```

## Usage

### 1. Preprocess Dataset
Ensure all images are in `RGBA` mode. 

### 2. Train and evaluate the Model
Run the training script to start training as well as evaluate the Faster R-CNN model:
```bash
python train.py
```

## File Descriptions
- **`train.py`**: Script to train and evaluate the Faster R-CNN model.
- **`requirements.txt`**: File listing the dependencies for the project.

## Contribution
Contributions are welcome! Feel free to submit a pull request or open an issue for discussion.

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [COCO API](https://github.com/cocodataset/cocoapi)
