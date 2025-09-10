import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO
import yt_dlp
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import time

st.title("👀 Person Detection in ROI 🎥")

# Load YOLO model
model = YOLO("yolov8s.pt")

# Play beep once per detection
def beep_once():
    if os.path.exists("alert.wav"):
        with open("alert.wav", "rb") as f:
            st.audio(f.read(), format="audio/wav", autoplay=True)

# Download YouTube video fully to temp file
def get_youtube_temp_video(yt_url):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": temp_file.name,
        "quiet": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])
    # Wait a moment to ensure file is written
    time.sleep(1)
    return temp_file.name

# Detect person in ROI
def detect_person_in_roi(frame, roi, model):
    detected = False
    if roi is None:
        return frame, detected
    x1, y1, x2, y2 = roi
    roi_frame = frame[y1:y2, x1:x2]
    results = model.predict(roi_frame, imgsz=640, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                detected = True
                xA, yA, xB, yB = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1+xA, y1+yA), (x1+xB, y1+yB), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1+xA, y1+yA-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, detected

# Select video source
source = st.radio("Select video source", ["Upload Video", "YouTube Link", "Live Camera"])
cap = None

# Uploaded video
if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_file.flush()
        cap = cv2.VideoCapture(temp_file.name)

# YouTube video
elif source == "YouTube Link":
    yt_url = st.text_input("Enter YouTube URL:")
    if yt_url:
        try:
            video_file = get_youtube_temp_video(yt_url)
            cap = cv2.VideoCapture(video_file)
        except Exception as e:
            st.error(f"Failed to load YouTube video: {e}")

# Live camera
elif source == "Live Camera":
    cap = cv2.VideoCapture(0)

roi = None

# Draw ROI on first frame
if cap:
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        st.write("🖌️ Draw a rectangle on the frame to select ROI")
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_color="red",
            background_image=pil_img,
            update_streamlit=True,
            height=frame_rgb.shape[0],
            width=frame_rgb.shape[1],
            drawing_mode="rect",
            key="roi_canvas"
        )
        if canvas.json_data and len(canvas.json_data["objects"]) > 0:
            obj = canvas.json_data["objects"][0]
            left, top, width, height = int(obj["left"]), int(obj["top"]), int(obj["width"]), int(obj["height"])
            roi = (left, top, left + width, top + height)
            st.success(f"ROI selected: {roi}")
        else:
            st.warning("No ROI selected. Detection will cover full frame.")

    stframe = st.empty()
    beep_triggered = False  # play beep once per detection

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if roi:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
        frame, detected = detect_person_in_roi(frame, roi, model)
        if detected and not beep_triggered:
            beep_once()
            beep_triggered = True
        elif not detected:
            beep_triggered = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

st.markdown(
    "<div style='position: fixed; bottom: 10px; right: 10px; font-size:12px;'>I have used ChatGPT to enhance project ❤️😊</div>",
    unsafe_allow_html=True
)
