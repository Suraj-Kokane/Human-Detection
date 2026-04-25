# Lightweight Real-Time Person Detection System

A lightweight, efficient person detection system optimized for edge devices and IoT applications. It uses YOLOv8 for object detection and implements **temporal smoothing** to provide a stable, flicker-free binary signal (True/False) that indicates human presence.

## Features

- **Real-Time Detection**: Captures live video from the webcam.
- **Lightweight Inference**: Uses YOLOv8 nano (`yolov8n.pt`) and low-resolution inference (320x320) to maintain high FPS (15-30) on CPU.
- **Temporal Smoothing (Anti-Flicker)**: The output signal switches to `True` instantly upon detection, but will only switch to `False` if a person is continuously absent for a 2-second threshold.
- **FastAPI Backend**: Provides API endpoints for `/start`, `/stop`, `/status`, and a `/video_feed` stream.
- **React Dashboard (Optional)**: A sleek dark-mode UI to control the camera, view the live video feed, and monitor the binary signal state in real-time.
- **Automated Logging**: Changes in occupancy state are automatically written to `utils/detection_log.txt`.

## Prerequisites

- Python 3.9+
- Node.js (if running the frontend)
- A connected webcam

## Project Structure

```text
person_detection/
├── backend/
│   └── main.py          # FastAPI application & API Endpoints
├── utils/
│   └── detector.py      # YOLOv8 integration and Stability Logic
├── models/
│   └── yolov8n.pt       # YOLOv8 weights (downloaded automatically)
├── frontend/
│   └── ...              # React + Vite dashboard
├── requirements.txt     # Python dependencies
└── README.md
```

## Running the Project

### 1. Start the Backend

Open a terminal in the `person_detection` directory:

```bash
# Optional: Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The backend is now accessible at `http://localhost:8000`.

### 2. Start the Frontend (Optional)

Open a new terminal in the `person_detection/frontend` directory:

```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

Navigate to the URL provided by Vite (usually `http://localhost:5173`) in your browser to access the dashboard.

## API Reference

- `GET /start`: Turns on the camera and starts processing frames.
- `GET /stop`: Stops processing and releases the camera.
- `GET /status`: Returns a JSON response containing the stable binary output: `{"person_present": true/false}`.
- `GET /video_feed`: A Multipart-JPEG stream of the processed video feed with bounding boxes and overlay text.

## Edge Device Optimizations Used

- **Reduced Image Size**: YOLOv8 inference runs at `imgsz=320` instead of the default 640.
- **Camera Resolution Limiting**: Standardized webcam capture resolution to 640x480.
- **Class Filtering**: Only looks for class `0` (person) and ignores all other 79 COCO dataset objects.
- **Temporal Thresholding**: Avoids false negatives during temporary occlusions or bounding box flickering.
