# AI-Fire-Detection-Prototype


<img width="1274" height="749" alt="Screenshot 2025-09-15 134817" src="https://github.com/user-attachments/assets/61e2ac06-1deb-4df3-bdaf-7a6863e89cf0" />


### Real-time Fire Detection Prototype
Real-time fire detection system that generates a dashboards for 10 cameras simultaneously, detects a fire, and  automatically sends an alert that indicates the timestamp, confidence and location (camera ID) of the fire.


### 🛠️  Installation

1. Install YOLOv8
```
pip install ultralytics
```
```
from ultralytics import YOLO

```

### 🏋️ Training

/Fire-Prototype.ipynb was set up for the training and deployment of the model. For training, a Kaggle fire dataset was downloaded and saved in the /data folder.

A .yaml file was created to read the data and use it for training.

+ .yaml file
    ```
    yaml_content = """
    train: data/train/images
    val:   data/valid/images
    test:  data/test/images


    nc: 1
    names: ['fire']
    """
    ```
    ```
    with open("fire.yaml", "w") as f:
        f.write(yaml_content)
    ```

A  YOLOv8 nano model was chosen as the base model and  fine-tuned using the fire dataset. The model was trained with 25 epochs of 16 image batches with imgsz 640.

+ Train
    ```
    model = YOLO("yolov8n.pt")
    results = model.train(
    data = "fire.yaml",
    epochs = 100,
    imgsz = 640,
    batch = 16
)


Due to low accuracy and undesirable results, the model was further trained with 100 epochs.
```
model = YOLO("runs/detect/train6/weights/best.pt")
results = model.train(
    data = "fire.yaml",
    epochs = 100,
    imgsz = 640,
    batch = 16
)
```



The results and their comparisons will be shown below.


### ⏱️ Results

25 epochs          |  100 epochs
:-------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------:
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/94fb4368-b812-45eb-ae08-45bbdb3ce32d" /> | <img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/1608ca5c-026a-46e7-b90b-9983ef55e73e" /> 



### 🔥 YOLOv8 Training Results

| Metric              | Description                                        | 25 Epochs (Old) | 100 Epochs (New) |
|----------------------|----------------------------------------------------|----------------|-----------------|
| **train/box_loss**    | Bounding box localization loss (lower = better)     | ~1.6–2.0        | **~0.8–1.0**     |
| **train/cls_loss**    | Classification loss (lower = better)                | ~3.0            | **~1.0**         |
| **train/dfl_loss**    | Distribution focal loss (box precision)             | ~1.6             | **~1.0**         |
| **val/box_loss**      | Validation bounding box loss                         | ~2.2             | **~1.8**         |
| **val/cls_loss**      | Validation classification loss                       | ~3.8–4.0         | **~2.0**         |
| **val/dfl_loss**      | Validation dfl loss                                  | ~2.1              | **~1.8**         |
| **metrics/precision(B)** | % of predicted detections that were correct       | ~0.9              | **~0.8**         |
| **metrics/recall(B)**    | % of actual objects correctly detected             | ~0.3              | **~0.5**         |
| **metrics/mAP50(B)**     | Avg precision @ IoU 0.5 (standard accuracy)        | ~0.4              | **~0.5**         |
| **metrics/mAP50-95(B)**  | Avg precision @ IoU 0.5–0.95 (stricter accuracy)    | ~0.20              | **~0.27**        |

