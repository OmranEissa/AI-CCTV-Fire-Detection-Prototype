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



<img alt="results6" title="25 epochs" src="https://github.com/user-attachments/assets/225af339-8be2-4f8a-8ab7-8e92e5750731"/>. <img alt="results7" title= "100 epochs" src="https://github.com/user-attachments/assets/3d8da399-2fb8-4801-9fa1-2912377ca7f4"/>


### 25 epochs $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ 100 epochs

