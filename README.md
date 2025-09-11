# 📹 SmartROI-Surveillance  

Real-time **person detection in a user-defined Region of Interest (ROI)** using **Streamlit** and **YOLOv8**.  
Supports **video upload, YouTube streams, and live camera**. Draw ROI, detect people, and get alarm notifications.  
Interactive, user-friendly, and built with **Python** and **OpenCV**.  

---

## 🎥 Demo  
https://github.com/user-attachments/assets/e50187ea-598f-4b49-8c20-45b758c4b8bf  

---

## 🚀 Features  

- ✅ **Real-time Object Detection** using YOLOv8/YOLOv11 (Ultralytics)  
- ✅ **Smart ROI Monitoring** – focus on specific areas of the video feed  
- ✅ **Web-based UI with Streamlit** – no complex setups, runs in your browser  
- ✅ **Custom Model Support** – bring your own YOLO model (`.pt` file)  
- ✅ **Screenshots & Logging** – capture frames automatically when detections occur  
- ✅ **Audio/Visual Alerts** for detected objects  
- ✅ **Lightweight Deployment** – works locally, in Docker, or on Streamlit Cloud  

---

## 📂 Project Structure  

SmartROI-Surveillance/
│ ├── streamlit_app.py
│ ├── yolov8n.pt
│ ├── yolov8s.pt
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .streamlit/ # Streamlit config (if deploying online)

yaml
Copy code

---

## ⚙️ Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/devthapallykarthikgoud/SmartROI-Surveillance.git
cd SmartROI-Surveillance
2️⃣ Create a Virtual Environment (Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Usage
Run Locally
bash
Copy code
streamlit run Main.py
This will start a local web app → open in your browser (default: http://localhost:8501).

Webcam Mode
By default, the system uses your webcam (cv2.VideoCapture(0)).
If you want to load a video file instead:

bash
Copy code
python Main.py --video video.mp4
Selecting YOLO Model
Choose between YOLOv3, YOLOv5, YOLOv8, YOLOv11 in the UI dropdown.

To add a custom model:

Place your .pt file inside models/

Select it from the UI dropdown

ROI (Region of Interest)
Inside the UI, you can:

✏️ Draw/select specific regions

🔔 Get alerts only if an object enters that ROI

Alerts & Screenshots
🔊 Audio alert plays once per detection

📸 Auto-screenshots are saved inside outputs/screenshots/

📑 Detection logs are saved as .csv inside outputs/logs/

🛠️ Advanced
Add New Objects/Faces
Use the "Add Object" button in the UI

Provide a label + reference image

System updates detection list dynamically

Deploy on Streamlit Cloud
Push repo to GitHub (already done ✅)

Connect repo to Streamlit Cloud

Add required secrets/config

App auto-deploys at:

arduino
Copy code
https://your-app.streamlit.app
📦 Requirements
Main dependencies (included in requirements.txt):

opencv-python

streamlit

ultralytics (for YOLO models)

pygame (for audio alerts)

numpy, pandas

