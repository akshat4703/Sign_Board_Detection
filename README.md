# 🚦 Traffic Sign Detection & Violation System (YOLOv8)

This project implements a **Traffic Sign Detection and Violation Monitoring System** using **YOLOv8** and **Streamlit**.
It detects traffic signs such as **Speed Limit, Stop, No Parking, Turn Left, and Turn Right**, and determines possible driving violations based on vehicle behavior.

---

## 📁 Project Structure

SIGN_BOARD_DETECTION/
│
├── runs/
│ └── detect/
│ ├── speed_limit_model/
│ ├── speed_limit_model2/
│ ├── train/
│ └── train2/
│
├── data.yaml
├── train.py
├── inference.py
├── yolov8s.pt
├── yolov11n.pt
└── README.md

---

## ✨ Features

- 🚘 Traffic sign detection using **YOLOv8**
- ⚠️ Violation detection for:
  - Over Speeding
  - Stop Sign Violation
  - No Parking Violation
  - Turn Restriction Violations (Left / Right / U-Turn)
- 🖼️ Image & 🎥 Video input support
- 📊 Annotated output with violation alerts
- 🌐 Interactive **Streamlit Web App**

---

## 🧠 Supported Traffic Signs

- Speed Limit Signs
- Stop
- No Parking
- Turn Left
- Turn Right

(Classes are configured via `data.yaml`)

---

## ⚙️ Requirements

Install dependencies using:

pip install ultralytics streamlit opencv-python pillow pyyaml numpy
Python 3.9+ recommended.

🏋️ Training the Model

Edit paths if required and run:
python train.py

Uses yolov8s.pt as base model

Trained results saved to:
runs/detect/speed_limit_model/

🚀 Running the Streamlit App
streamlit run inference.py


Then open the browser URL shown in the terminal.

🖼️ Image Inference

Upload an image
Set:
- Vehicle speed
- Vehicle motion
- Manoeuvre type
View annotated image
Download results

🎥 Video Inference

Upload video file
Real-time violation detection
Download processed video
View detected sign statistics

🧪 Model Files

yolov8s.pt – Base YOLOv8 model
best.pt – Trained model (inside runs/detect)

👨‍💻 Author
Akshat Pal