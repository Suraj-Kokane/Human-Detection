from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import threading
import sys
import os

# Ensure the utils module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.detector import PersonDetector

app = FastAPI(title="Real-Time Person Detection System")

# Enable CORS so the React frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector with stability logic parameters
detector = PersonDetector(confidence_threshold=0.5, stable_delay=2.0)

# Global variables to handle the camera state and concurrency
camera = None
camera_lock = threading.Lock()
is_running = False

def get_camera():
    """Initializes and returns the OpenCV VideoCapture object."""
    global camera
    if camera is None:
        # 0 is usually the default webcam
        camera = cv2.VideoCapture(0)
        # Set low resolution to optimize CPU performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Try to cap FPS at 30
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def release_camera():
    """Releases the camera resource safely."""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    """
    Generator function that continuously reads frames from the camera,
    processes them through the YOLO detector, and yields JPEG images.
    """
    global is_running, camera
    
    cap = get_camera()
    
    while is_running:
        success, frame = cap.read()
        if not success:
            break
            
        # Process the frame for person detection
        processed_frame, _ = detector.process_frame(frame)
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield the output frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    # Clean up when stopped
    release_camera()

@app.get("/start")
def start_detection():
    """Starts the real-time camera processing loop."""
    global is_running
    with camera_lock:
        if not is_running:
            is_running = True
            return {"status": "Detection started", "running": True}
        return {"status": "Detection is already running", "running": True}

@app.get("/stop")
def stop_detection():
    """Stops the camera processing loop and releases the camera."""
    global is_running
    with camera_lock:
        if is_running:
            is_running = False
            return {"status": "Detection stopped", "running": False}
        return {"status": "Detection is not running", "running": False}

@app.get("/status")
def get_status():
    """
    Returns the current stable binary signal (True/False).
    Used by IoT edge devices or the frontend dashboard to check occupancy.
    """
    return {
        "person_present": detector.is_person_present,
        "is_running": is_running
    }

@app.get("/video_feed")
def video_feed():
    """
    Endpoint that provides a continuous multipart image stream (MJPEG).
    """
    if not is_running:
        return JSONResponse(status_code=400, content={"error": "Camera is not running. Call /start first."})
        
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
