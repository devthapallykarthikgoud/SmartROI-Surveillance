import streamlit as st
import tempfile
import cv2
import platform
import os
from ultralytics import YOLO

if platform.system() == "Windows":
    import winsound
else:
    from playsound import playsound

def beep():
    if platform.system() == "Windows":
        winsound.Beep(1000, 500)
    else:
        if os.path.exists("alert.wav"):
            playsound("alert.wav")

st.set_page_config(page_title="Smart ROI Surveillance", layout="wide")
st.title("Smart ROI Surveillance")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
use_cam = st.checkbox("Use Webcam")

conf = st.slider("Confidence", 0.1, 1.0, 0.5)
model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])
model = YOLO(model_name)

if video_file or use_cam:
    tmp = None
    if video_file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(video_file.read())
        path = tmp.name
    else:
        path = 0

    cap = cv2.VideoCapture(path)
    frame_window = st.empty()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, conf=conf)
        view = results[0].plot()
        frame_window.image(view, channels="BGR")

        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name.lower() == "person":
                beep()
                break

    cap.release()
    if tmp:
        os.remove(tmp.name)
