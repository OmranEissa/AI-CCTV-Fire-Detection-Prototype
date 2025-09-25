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
    with open("fire1.yaml", "w") as f:
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
    ```




### ⏱️ Results

#### Prediction Results  

| Ground Truth       | Prediction 
|--------------------|-----------------
|![val_batch1_labels](https://github.com/user-attachments/assets/314c9b90-22fa-46fb-b030-6122d4583314) | ![val_batch1_pred](https://github.com/user-attachments/assets/2f45d8b4-d67b-40eb-8b42-a3dcb8b378a0)





| BoxF1 Curve  | BoxP Curve    
|--------------------------|--------------------------------------------------------
|   <img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/4cf8f32e-bf63-4109-b1ea-2303da04a029" /> |   <img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/b605d32e-f574-4b45-931e-67109d0dbfca" /> 



|BoxPR Curve      | BoxR Curve
|-----------------|---------------
| <img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/5542e4af-aefb-4a53-ba69-ba0e235e3343" /> |<img width="2250" height="1500" alt="BoxR_curve" src="https://github.com/user-attachments/assets/2858f934-ffc2-4b0a-b640-f82e67db16b8" />


|Metrics 
|--------
|<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/a86a5f24-6285-4bce-8110-1bc3888cc684" />




### 🖥️ Code 

#### This is where we implement the CCTV part of the project and the alert system and the dashboard on which the CCTV cameras will be displayed.

```
import cv2, numpy as np, time, json
from ultralytics import YOLO

# Video inputs
sources = [
    "CCTV1.mp4","CCTV2.mp4","CCTV3.mp4",
    "CCTV4.mp4","CCTV5.mp4","CCTV6.mp4", "CCTV7.mp4","CCTV8.mp4",
    "CCTV9.mp4", "CCTV10.mp4"
]
caps = [cv2.VideoCapture(src) for src in sources]
model = YOLO("runs/detect/train4/weights/best.pt")


frame_skip = 1
frame_count = 0
target_size = (320, 240)  

#Resizing window to viewable size.
cv2.namedWindow("Control Room", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control Room", 1280, 720)

#Function to log and store alerts.
def AlertLog(cam_id, cls, conf, bbox, ts):
    alert = {"camera":cam_id+1,"class":cls,"conf":round(conf,3),
             "bbox":bbox,"timestamp":ts}
    with open("alerts.jsonl","a") as f: f.write(json.dumps(alert)+"\n")
    print("ALERT:", alert)

while True:
    frames = []
    frame_count += 1

    for cam_id, cap in enumerate(caps):
        ret, frame = cap.read()

        # Looping video
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if not ret or frame is None:
            # If camera is offline
            frame = np.zeros((target_size[1], target_size[0], 3), np.uint8)
            cv2.putText(frame, f"Camera {cam_id +1} Offline", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            frame = cv2.resize(frame, target_size)
            cv2.putText(frame, f"Cam {cam_id +1 }", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if frame_count % frame_skip == 0:
                results = model.predict(frame, conf=0.25, verbose=False)
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf >= 0.80:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            cls = r.names[cls_id]
                            ts = time.strftime("%Y-%m-%d %H:%M:%S")
                            AlertLog(cam_id, cls, conf, [x1,y1,x2,y2], ts)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            cv2.putText(frame, f"{cls} {conf:.2f}",
                                        (x1, max(0, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (0, 0, 255), 1)

        frames.append(frame)

    # grid
    row1 = np.hstack(frames[:5])
    row2 = np.hstack(frames[5:])
    grid = np.vstack([row1, row2])

    # scale to fit screen
    display_grid = cv2.resize(grid, (1280, 720), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Control Room", display_grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

for cap in caps: cap.release()
cv2.destroyAllWindows()

```
### Alerts and Dashboard
#### This is what the Dashboard looks like with all 10 cameras and the logged alerts.

|Dashboard   
|-------------
| <img width="1274" height="749" alt="Screenshot 2025-09-15 134817" src="https://github.com/user-attachments/assets/02ba2298-d706-4991-8766-7254e2012dcc" /> 


|Alerts
|-------
|<img width="952" height="320" alt="Screenshot 2025-09-21 110751" src="https://github.com/user-attachments/assets/39cc34e4-7715-46e1-8c82-15221f5229ba" />


### 🔗 References

+ <https://github.com/spacewalk01/yolov5-fire-detection?tab=readme-ov-file#readme>
+ <https://www.youtube.com/watch?v=-RDeVPHipZU>
+ <https://www.youtube.com/watch?v=FBavXyN18K8&list=PL4Cc4cDq3t9mCdZ3t0czemfz7VwPdSjj9>
