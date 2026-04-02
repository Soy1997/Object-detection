# YOLOv7 Object Detection System
<img width="1391" height="790" alt="Screenshot 2026-04-02 at 10 17 44 AM" src="https://github.com/user-attachments/assets/90baebf2-2e59-4d31-a0e8-eec0df176898" />

<img width="1393" height="804" alt="Screenshot 2026-04-02 at 10 16 35 AM" src="https://github.com/user-attachments/assets/a60c9286-3327-413e-be2e-29eaffc0d4bd" />


Real-time and static object detection powered by **YOLOv7**, **OpenCV**, and **Flask** — available as standalone Python scripts or a full browser-based web app.


##  Features

-  **Image Detection** — Upload any image and detect objects instantly
-  **Real-Time Webcam Detection** — Live object detection via your webcam
-  **Web Interface** — Sci-fi themed browser UI (no command line required)
-  **Fast Inference** — Single forward pass detection with YOLOv7
-  **Smart Model Caching** — Model loads once, reused across all requests
-  **Secure File Handling** — Path traversal protection on all uploads
-  **80 Object Classes** — Detects people, cars, animals, furniture, and more (COCO dataset)


## Project Structure

```
├── 01_ImageObjectDetection.py      # Standalone image detection script
├── 03_RealTimeObjectDetection.py   # Standalone webcam detection script
├── app.py                          # Flask web server
├── templates/
│   ├── index.html                  # Homepage — mode selection
│   ├── webcam.html                 # Live webcam feed page
│   └── result.html                 # Detection results page
├── weights/
│   └── yolov7.weights              # Pre-trained model weights
├── cfg/
│   └── yolov7.cfg                  # Model architecture config
└── coco.names                      # 80 class labels
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/yolov7-object-detection.git
cd yolov7-object-detection
```

### 2. Install dependencies

```bash
pip install opencv-python flask numpy werkzeug
```

### 3. Download YOLOv7 weights

Download `yolov7.weights` and place it in the `weights/` directory.

> The weights file contains the pre-trained neural network — this is where all the detection intelligence lives.

---

##  Usage

### Option A — Web App (Recommended)

```bash
python app.py
```

Then open your browser at `http://localhost:5000`. Choose between image upload or live webcam detection from the homepage.

### Option B — Standalone Scripts

**Detect objects in a single image:**
```bash
python 01_ImageObjectDetection.py
```

**Real-time webcam detection:**
```bash
python 03_RealTimeObjectDetection.py
# Press ESC to stop
```

---

##  How It Works

### Detection Pipeline (5 Steps)

Every detection — whether image or webcam — runs through the same pipeline:

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | **Load Model** | YOLOv7 weights loaded into OpenCV's DNN module |
| 2 | **Blob Conversion** | Image normalized (0–1 range), resized to 320×320, BGR→RGB |
| 3 | **Forward Pass** | Single pass through the neural network returns raw predictions |
| 4 | **Confidence Filter** | Detections below 30% confidence are discarded |
| 5 | **NMS** | Non-Maximum Suppression removes duplicate overlapping boxes |

### Why YOLO?

Traditional object detectors scan an image in multiple passes. YOLO (**You Only Look Once**) processes the **entire image in a single forward pass**, making it dramatically faster and suitable for real-time applications.

### Flask Web App — Key Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Homepage — mode selection |
| `/image_detection` | POST | Accepts uploaded image, runs detection |
| `/predictions/<filename>` | GET | Displays annotated result image |
| `/video_feed` | GET | Streams live webcam frames as MJPEG |
| `/webcam_feed` | POST | Opens webcam detection page |

### Live Webcam Streaming

The webcam feed uses **MJPEG streaming** — a generator function continuously yields JPEG frames, and the browser treats it as a continuous video stream via a plain `<img>` tag. No WebSockets or JavaScript required.

---

##  Tech Stack

| Library | Role |
|---------|------|
| `opencv-python` | Image I/O, blob creation, DNN inference, bounding box drawing |
| `numpy` | Array math for scores, boxes, and color generation |
| `flask` | Web server, routing, HTML templating, file handling |
| `werkzeug` | `secure_filename()` for safe file uploads |
| `functools.lru_cache` | Caches the loaded YOLO model across requests |
| `pathlib` | Cross-platform file path handling |

---

##  Configuration

**Confidence threshold** — controls detection sensitivity (default: `0.3`):
```python
if conf > 0.3:  # Increase to reduce false positives, decrease to catch more objects
```

**NMS thresholds** — controls duplicate suppression (default: `0.5, 0.4`):
```python
indexes = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.5, nms_threshold=0.4)
```

**Webcam source** — change `0` to use a different camera or video file:
```python
cap = cv2.VideoCapture(0)
```

---

##  Model Files

| File | Description |
|------|-------------|
| `yolov7.weights` | Trained neural network weights (~70MB) |
| `yolov7.cfg` | Model architecture definition |
| `coco.names` | 80 class labels (person, car, dog, bottle, etc.) |

> **Note:** Only the `.weights` file needs to be downloaded separately. The `.cfg` and `.names` files are included in this repo.

---

##  License

This project uses YOLOv7, which is subject to its own license. See the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) for details.
