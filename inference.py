import streamlit as st
from ultralytics import YOLO
import tempfile, os, time, io
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import yaml
from collections import Counter

st.set_page_config(page_title="Traffic Sign Violation (YOLO)", layout="wide")

# ================== PATH CONFIG ==================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.yaml"

# ================== USER-REQUESTED HARD-CODED MODEL PATH ==================

MODEL_PATH = r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\sign_board_detection\runs\detect\speed_limit_model\weights\best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ best.pt not found at: {MODEL_PATH}")
    st.stop()

selected_model_path = MODEL_PATH
st.success(f"Using model: {selected_model_path}")

CONF_DEFAULT = 0.4
IMG_SIZE_DEFAULT = 640
LIMIT_HOLD_SEC = 5.0

# ================== HELPERS ==================

@st.cache_resource
def load_model_safe(path: str):
    return YOLO(path)

@st.cache_resource
def load_class_names(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return names

@st.cache_resource
def build_speed_limit_mapping(class_names: dict):
    mapping = {}
    for cid, name in class_names.items():
        if not isinstance(name, str):
            continue
        low = name.lower()
        if "speed limit" in low and any(ch.isdigit() for ch in low):
            parts = low.replace("/", " ").split()
            nums = [p for p in parts if p.isdigit()]
            if nums:
                mapping[cid] = float(nums[-1])
    return mapping

def classify_turn_sign(cls_name_lower: str):
    if "no u-turn" in cls_name_lower or "no u turn" in cls_name_lower:
        return "no_uturn"
    if "no left" in cls_name_lower:
        return "no_left"
    if "no right" in cls_name_lower:
        return "no_right"
    return None


# ================== PROCESS FRAME ==================

def process_frame_with_violations(
    model,
    frame_bgr,
    class_names,
    speed_limit_classes,
    vehicle_speed,
    car_moving,
    manoeuvre,
    last_limit_value,
    last_limit_time,
    conf,
    imgsz,
):

    results = model.predict(frame_bgr, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]

    vis = frame_bgr.copy()
    h, w, _ = vis.shape

    any_violation = False
    violated_msgs = []
    detected_signs = []

    current_limit = None
    best_conf = 0

    man = manoeuvre.lower()
    is_parked = "parked" in man
    turning_left = "turning left" in man
    turning_right = "turning right" in man
    doing_uturn = "u-turn" in man or "u turn" in man

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf_box = float(box.conf[0])
            cls_name = class_names.get(cls, str(cls))
            cls_low = cls_name.lower()

            detected_signs.append(cls_name)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{cls_name} {conf_box*100:.1f}%"
            cv2.putText(vis, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

            # SPEED LIMIT
            if cls in speed_limit_classes:
                limit_val = speed_limit_classes[cls]
                if conf_box > best_conf:
                    best_conf = conf_box
                    current_limit = limit_val
                if vehicle_speed > limit_val + 2:
                    any_violation = True
                    violated_msgs.append(f"Over Speed {int(limit_val)}")

            # RED LIGHT
            if cls_low == "red light" and car_moving:
                any_violation = True
                violated_msgs.append("Red Light Violation")

            # STOP SIGN
            if cls_low == "stop" and car_moving:
                any_violation = True
                violated_msgs.append("Stop Sign Violation")

            # NO PARKING
            if "no parking" in cls_low and is_parked:
                any_violation = True
                violated_msgs.append("No Parking Violation")

            # TURN RESTRICTIONS
            t = classify_turn_sign(cls_low)
            if t == "no_left" and turning_left:
                any_violation = True
                violated_msgs.append("No Left Turn Violation")
            if t == "no_right" and turning_right:
                any_violation = True
                violated_msgs.append("No Right Turn Violation")
            if t == "no_uturn" and doing_uturn:
                any_violation = True
                violated_msgs.append("No U-Turn Violation")

    # speed limit memory
    now = time.time()
    if current_limit is not None:
        last_limit_value = current_limit
        last_limit_time = now
    elif last_limit_value and now - last_limit_time > LIMIT_HOLD_SEC:
        last_limit_value = None

    # HUD
    cv2.putText(vis, f"Speed: {vehicle_speed:.1f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),2)

    if last_limit_value is not None:
        cv2.putText(vis, f"Limit: {int(last_limit_value)}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)

    # violation
    if any_violation:
        cv2.rectangle(vis, (0,0), (w,70), (0,0,255), -1)
        cv2.putText(vis, "SIGN VIOLATED", (20,50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255,255,255),3)

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), last_limit_value, last_limit_time, any_violation, violated_msgs, detected_signs


# ================== VIDEO INFERENCE (WINDOWS SAFE) ==================

def infer_video_with_violations(
    model,
    video_path,
    class_names,
    speed_limit_classes,
    vehicle_speed,
    car_moving,
    manoeuvre,
    conf,
    imgsz,
    progress_callback=None,
):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FIXED — SAFE TEMP VIDEO
    out_path = os.path.join(tempfile.gettempdir(), f"processed_{int(time.time())}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_limit_value = None
    last_limit_time = 0
    sign_counter = Counter()
    frame_idx = 0
    preview = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        annotated, last_limit_value, last_limit_time, any_v, msgs, signs = process_frame_with_violations(
            model, frame, class_names, speed_limit_classes, vehicle_speed,
            car_moving, manoeuvre, last_limit_value, last_limit_time,
            conf, imgsz
        )

        sign_counter.update(signs)
        writer.write(cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))

        if progress_callback:
            progress_callback(frame_idx / total)

        if frame_idx % 10 == 0:
            preview = annotated.copy()

    cap.release()
    writer.release()

    return out_path, frame_idx, preview, sign_counter


# ================== STREAMLIT UI ==================

st.title("🚦 Traffic Sign Detection & Violation (YOLO)")

if not DATA_PATH.exists():
    st.error("data.yaml not found!")
    st.stop()

class_names = load_class_names(DATA_PATH)
speed_limit_classes = build_speed_limit_mapping(class_names)


# Load model
with st.spinner("Loading model..."):
    model = load_model_safe(selected_model_path)


# Controls
col1, col2 = st.columns([1,2])

with col1:
    conf = st.slider("Confidence", 0.1, 1.0, CONF_DEFAULT)
    imgsz = st.selectbox("Image Size", [320,416,512,640], index=3)
    vehicle_speed = st.slider("Vehicle Speed", 0, 160, 50)
    car_moving = st.checkbox("Vehicle Moving", True)
    manoeuvre = st.selectbox("Manoeuvre",
        ["Driving straight","Parked","Turning left","Turning right","U-turn"])

with col2:
    uploaded = st.file_uploader("Upload Image / Video",
        type=["jpg","jpeg","png","bmp","mp4","mov","avi","mkv"])


# ================== HANDLE UPLOAD ==================

if uploaded:
    ext = uploaded.name.split(".")[-1].lower()
    data = uploaded.read()

    # IMAGE
    if ext in ["jpg","jpeg","png","bmp"]:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        annotated, _, _, any_v, msgs, signs = process_frame_with_violations(
            model, bgr, class_names, speed_limit_classes, vehicle_speed,
            car_moving, manoeuvre, None, 0, conf, imgsz
        )

        st.image(annotated)
        if any_v:
            st.error("SIGN VIOLATION: " + ", ".join(set(msgs)))
        else:
            st.success("No violations detected.")

        buf = io.BytesIO()
        annotated.save(buf, format="JPEG")
        st.download_button("Download Annotated Image", buf.getvalue(),
                           file_name="annotated.jpg", mime="image/jpeg")

    # VIDEO
    else:
        # SAFE VIDEO PATH
        video_path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(video_path, "wb") as f:
            f.write(data)

        progress = st.progress(0.0)

        out_path, frames, preview, stats = infer_video_with_violations(
            model, video_path, class_names, speed_limit_classes,
            vehicle_speed, car_moving, manoeuvre, conf, imgsz,
            progress_callback=lambda x: progress.progress(x)
        )

        st.video(out_path)

        with open(out_path, "rb") as f:
            st.download_button(
                "Download Annotated Video",
                data=f.read(),
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )

        if stats:
            st.json(dict(stats))
