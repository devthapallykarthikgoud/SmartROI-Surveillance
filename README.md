## SmartROI-Surveillance

Real-time person detection in a user-defined Region of Interest (ROI) using Streamlit and YOLOv8. Supports video upload, YouTube streams, and live camera. Draw ROI, detect people, and get alarm notifications. Interactive, user-friendly, and built with Python and OpenCV.


https://github.com/user-attachments/assets/e50187ea-598f-4b49-8c20-45b758c4b8bf

📹 SmartROI-Surveillance

SmartROI-Surveillance is an advanced AI-powered smart surveillance system built using YOLO (You Only Look Once) and OpenCV.
It allows real-time object detection, region-of-interest (ROI) monitoring, alerts, and live streaming via Streamlit.

🚀 Features

✅ Real-time Object Detection using YOLOv8/YOLOv11 (Ultralytics).

✅ Smart ROI (Region of Interest) Monitoring – focus on specific areas of the video feed.

✅ Web-based UI with Streamlit – no complex setups, just run in your browser.

✅ Custom Model Support – bring your own YOLO model .pt file.

✅ Screenshots & Logging – capture frames automatically when detections occur.

✅ Audio/Visual Alerts for detected objects.

✅ Lightweight Deployment – works locally, in Docker, or on Streamlit Cloud.

📂 Project Structure
SmartROI-Surveillance/
│── Main.py                  # Entry point for the app
│── ObjectDetection.py       # YOLO-based detection logic
│── AudioManager.py          # Audio alert handling
│── utils.py                 # Helper functions
│
├── models/                  # Pre-trained YOLO configs/weights
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   ├── yolov3-labels.txt
│
├── assets/                  # Audio files, icons, sample images
├── outputs/                 # Auto-saved screenshots and logs
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .streamlit/              # Streamlit config (if deploying online)

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/devthapallykarthikgoud/SmartROI-Surveillance.git
cd SmartROI-Surveillance

2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Usage
Run Locally
streamlit run Main.py


This will start a local web app → open in your browser (default: http://localhost:8501).

Webcam Mode

By default, the system uses your webcam (cv2.VideoCapture(0)).
If you want to load a video file instead:

python Main.py --video video.mp4

Selecting YOLO Model

You can choose different YOLO versions (v3, v5, v8, v11) inside the UI dropdown.
To add a custom model:

Place your .pt file inside models/.

Select it from the UI dropdown.

ROI (Region of Interest)

Inside the UI, you can:

Draw/select specific regions.

Get alerts only if an object enters that ROI.

Alerts & Screenshots

Audio alert plays once per detection.

Auto-screenshots are saved inside outputs/screenshots/.

Detection logs are saved as .csv inside outputs/logs/.

🛠️ Advanced
Add New Objects/Faces

Use the "Add Object" button in UI.

Provide a label + reference image.

System will retrain/update detection list dynamically.

Deploy on Streamlit Cloud

Push repo to GitHub (already done ✅).

Connect repo to Streamlit Cloud
.

Add required secrets/config.

It will auto-deploy at https://your-app.streamlit.app.

📦 Requirements

Main dependencies:

opencv-python

streamlit

ultralytics (for YOLO models)

pygame (for audio alerts)

numpy, pandas (for data handling)

All included in requirements.txt.

🎯 Example Demo


(Put a sample demo GIF/video here so users see it in action.)

🤝 Contributing

Pull requests are welcome!

Fork the repo

Create a feature branch

Submit a PR
