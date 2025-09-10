import streamlit as st
import tempfile
import cv2
import platform
import os
from ultralytics import YOLO

# Audio cross-platform handling
if platform.system() == "Windows":
    import winsound
else:
    from playsound import playsound


# ========================
# Utility: Play Sound
# ========================
def play_alert():
    """Play alert sound cross-platform"""
    if platform.system() == "Windows":
        winsound.Beep(1000, 500)  # freq=1000Hz, duration=500ms
    else:
        # Expect an alert.wav in root folder
        if os.path.exists("alert.wav"):
            playsound("alert.wav")
        else:
            st.warning("Alert sound file not found (alert.wav)")


# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="Smart ROI Surveillance", layout="wide")
st.title("🎥 Smart ROI Surveillance with YOLOv8")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
use_webcam = st.checkbox("Use Webcam instead of video")

confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

model_choice = st.selectbox("Choose YOLO model", ["yolov8n.pt", "yolov8s.pt"])
model = YOLO(model_choice)

# ========================
# Video Processing
# ========================
if uploaded_file or use_webcam:
    tfile = None
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    else:
        video_path = 0  # webcam

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot()

        # Show frame in Streamlit
        stframe.image(annotated_frame, channels="BGR")

        # Example: play alert if person detected
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label.lower() == "person":
                play_alert()
                break  # avoid multiple alerts per frame

    cap.release()
    if tfile:
        os.remove(tfile.name)
