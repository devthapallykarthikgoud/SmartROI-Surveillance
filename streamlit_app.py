import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO
import yt_dlp
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import platform
import os
import simpleaudio as sa

st.set_page_config(page_title="Smart ROI Surveillance", layout="wide")
st.title("👀 Smart ROI Person Detection 🎥")

model = YOLO("yolov8s.pt")

def get_youtube_stream_url(yt_url):
    ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_url, download=False)
        return info["url"]

def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 300)
    else:
        if os.path.exists("alert.wav"):
            try:
                sa.WaveObject.from_wave_file("alert.wav").play()
            except Exception:
                pass

def detect_person_in_roi(frame, roi, model):
    found = False
    if roi is None:
        return frame, found
    x1, y1, x2, y2 = roi
    roi_frame = frame[y1:y2, x1:x2]
    results = model.predict(roi_frame, imgsz=640, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                found = True
                xA, yA, xB, yB = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1+xA, y1+yA), (x1+xB, y1+yB), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1+xA, y1+yA-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, found

source = st.radio("Video source", ["Upload Video", "YouTube Link", "Live Camera"])
cap = None

if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file.name)

elif source == "YouTube Link":
    yt_url = st.text_input("Enter YouTube URL")
    if yt_url:
        try:
            stream_url = get_youtube_stream_url(yt_url)
            cap = cv2.VideoCapture(stream_url)
        except Exception as e:
            st.error(f"Failed to load YouTube video: {e}")

elif source == "Live Camera":
    cap = cv2.VideoCapture(0)

roi = None
alert_playing = False

if cap:
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        st.write("🖌️ Draw ROI (Region of Interest)")
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
            left, top = int(obj["left"]), int(obj["top"])
            width, height = int(obj["width"]), int(obj["height"])
            roi = (left, top, left + width, top + height)
            st.success(f"ROI selected: {roi}")

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if roi:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
        frame, found = detect_person_in_roi(frame, roi, model)

        if found and not alert_playing:
            beep()
            alert_playing = True
        elif not found:
            alert_playing = False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

st.markdown(
    "<div style='position: fixed; bottom: 10px; right: 10px; font-size:12px;'>Made with ❤️ using ChatGPT</div>",
    unsafe_allow_html=True
)
