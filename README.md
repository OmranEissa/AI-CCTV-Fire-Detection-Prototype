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

A  YOLOv8 nano model was chosen as the base model and  fine-tuned using the fire dataset. The model was trained with 100 epochs of 8 image batches with imgsz 640. A large [dataset](https://universe.roboflow.com/situational-awarnessinnovsense/fire-detection-ypseh) of almost 10,000 images was used. The dataset can be found here 

+ Train
    ```
  model = YOLO("yolov8n.pt")
    results = model.train(
    data = "fire1.yaml",
    epochs = 100,
    imgsz = 640,
    batch = 8
)



### ⏱️ Results





| BoxF1 Curve  | BoxP Curve    
|--------------------------|--------------------------------------------------------
|   <img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/4cf8f32e-bf63-4109-b1ea-2303da04a029" /> |   <img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/b605d32e-f574-4b45-931e-67109d0dbfca" /> 
  |BoxPR Curve      | BoxR Curve

| <img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/5542e4af-aefb-4a53-ba69-ba0e235e3343" /> |<img width="2250" height="1500" alt="BoxR_curve" src="https://github.com/user-attachments/assets/2858f934-ffc2-4b0a-b640-f82e67db16b8" />
