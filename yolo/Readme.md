# YOLO Project

This project involves training and testing object detection models using the YOLO (You Only Look Once) architecture. The objective is to evaluate the performance of YOLO models on traffic sign detection tasks, particularly in adverse outdoor conditions. Additionally, synthetic data generated using the CARLA simulator is combined with real-world data to assess improvements in model robustness.

---

## **Project Overview**

- **Goal**: Evaluate the impact of using synthetic data for training object detection models, especially under challenging conditions like rain, fog, and night.
- **Datasets**:
  - **Real Images**: Collected from real-world sources.
  - **AI-Generated Images**: Generated using the CARLA simulator.
  - **Combined Dataset**: A mix of real and synthetic images.
- **Models**: YOLOv5 (trained with custom datasets).

---

## **Features**

1. **Traffic Sign Detection**: Detect various types of traffic signs in images.
2. **Adverse Condition Testing**: Evaluate model robustness in conditions like:
   - Sunny
   - Rain
   - Fog
   - Night
3. **Synthetic Data Augmentation**: Assess improvements in performance with AI-generated images.

---

## **Project Structure**

```
YOLO_Project/
├── datasets/
│   ├── real_images/
│   ├── ai_generated_images/
│   ├── combined_dataset/
├── models/
│   ├── yolov5/
│   └── weights/
├── outputs/
│   ├── predictions/
│   └── metrics/
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── utils.py
├── requirements.txt
└── README.md
```

---

## **Setup and Installation**

### Prerequisites

- Python 3.8+
- NVIDIA CUDA 10.2+ (for GPU acceleration)
- Git
- pip

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-URL>
   cd YOLO_Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the YOLO environment (if applicable):
   - Download YOLOv5 weights from the [official YOLO GitHub repository](https://github.com/ultralytics/yolov5).

4. Prepare datasets:
   - Place the datasets in the `datasets/` directory.

---

## **Usage**

### Training

Train the YOLO model with the desired dataset:
```bash
python scripts/train.py --data datasets/combined_dataset --epochs 100 --weights models/yolov5/weights/initial.pt
```

### Testing

Test the trained model on a specific dataset:
```bash
python scripts/test.py --data datasets/real_images --weights models/yolov5/weights/final.pt
```

### Metrics and Results

- The output metrics and predictions are saved in the `outputs/` directory.
- Confusion matrix and mAP scores are generated for each test run.

---


