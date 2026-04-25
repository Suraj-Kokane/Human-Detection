import cv2
import time
import os
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="models/yolov8n.pt", confidence_threshold=0.5, stable_delay=2.0):
        """
        Initialize the Person Detector with YOLOv8.
        
        :param model_path: Path to the YOLOv8 model weights.
        :param confidence_threshold: Minimum confidence score to consider a valid detection.
        :param stable_delay: Time in seconds before switching to False after losing the person.
        """
        # Ensure the model directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load the YOLOv8 model (downloads 'yolov8n.pt' automatically if not found locally)
        self.model = YOLO("yolov8n.pt") 
        self.confidence_threshold = confidence_threshold
        self.stable_delay = stable_delay
        
        # State tracking for stability logic
        self.last_detection_time = 0
        self.is_person_present = False
        
        # Setup log file
        self.log_file = "utils/detection_log.txt"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
    def _log_detection(self, status):
        """Logs when the stable status changes."""
        with open(self.log_file, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Person Present: {status}\n")

    def process_frame(self, frame):
        """
        Run inference on a single frame and apply temporal smoothing to the output signal.
        
        :param frame: Video frame from OpenCV.
        :return: Tuple of (Processed Frame with Bounding Boxes, Stable Signal State)
        """
        # Run inference. We use imgsz=320 for speed (CPU friendly)
        # We only look for class 0 (person)
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, imgsz=320, verbose=False)
        
        current_time = time.time()
        person_detected_in_frame = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class 0 corresponds to "person" in COCO dataset
                if int(box.cls[0]) == 0:
                    person_detected_in_frame = True
                    
                    # Draw bounding box and label
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        # Stability Logic: Temporal Smoothing
        # - If detected: set immediately to True and update last_detection_time
        # - If NOT detected: check if time since last detection > stable_delay (e.g. 2s)
        #   If so, switch to False. This prevents flickering.
        previous_status = self.is_person_present

        if person_detected_in_frame:
            self.last_detection_time = current_time
            self.is_person_present = True
        else:
            if current_time - self.last_detection_time > self.stable_delay:
                self.is_person_present = False
                
        # Log if status just changed
        if previous_status != self.is_person_present:
            self._log_detection(self.is_person_present)

        # Draw the stable signal status on the frame
        status_text = f"Stable Signal: {'True' if self.is_person_present else 'False'}"
        color = (0, 255, 0) if self.is_person_present else (0, 0, 255)
        
        # Display the stable output signal visually
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        return frame, self.is_person_present
