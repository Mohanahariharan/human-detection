from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow
import os
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
camera_active = False
cap = None
model_roboflow = None
model_yolo = None
USE_TEST_VIDEO = os.environ.get('RENDER', False)  # Auto-enable for cloud

def init_models():
    global model_roboflow, model_yolo
    try:
        # Initialize Roboflow model
        rf = Roboflow(api_key="vuAfslycGYT8XKOu4lId")
        project = rf.workspace().project("thermal-person-df3lf")
        model_roboflow = project.version(1).model
        logger.info("Roboflow model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Roboflow model: {str(e)}")
        model_roboflow = None

    try:
        # Initialize YOLO model with smaller version
        model_yolo = YOLO('yolov8n.pt').half()  # Use FP16 to reduce memory
        logger.info("YOLO model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {str(e)}")
        model_yolo = None

def init_camera():
    global cap
    if USE_TEST_VIDEO:
        # Use test video in cloud environment
        test_video = "test_thermal.mp4"
        if not os.path.exists(test_video):
            logger.error(f"Test video {test_video} not found!")
            return False
            
        cap = cv2.VideoCapture(test_video)
        if cap.isOpened():
            logger.info("Using test video feed")
            return True
        return False
    else:
        # Try local camera for development
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                logger.info(f"Camera opened at index {camera_index}")
                try:
                    print("camera successfully")
                except:
                    print("error 500")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 96)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
                return True
        logger.error("Could not open any camera")
        return False

def release_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None

def generate_frames():
    global camera_active, cap
    SCALE_FACTOR = 6
    
    # Create placeholder image
    placeholder = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Camera Off", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    while True:
        if not camera_active:
            # Show placeholder when camera is off
            enlarged = cv2.resize(placeholder, 
                                (96 * SCALE_FACTOR, 96 * SCALE_FACTOR),
                                interpolation=cv2.INTER_NEAREST)
            ret, buffer = cv2.imencode('.jpg', enlarged)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
            
        if cap is None or not cap.isOpened():
            if not init_camera():
                time.sleep(0.1)
                continue
        
        ret, frame = cap.read()
        if not ret:
            if USE_TEST_VIDEO:
                # Loop test video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.1)
            continue

        # Process frame
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.applyColorMap(frame_gray, cv2.COLORMAP_JET)

            # Run detections
            frame_rgb = run_detections(frame_rgb)
            
            # Resize for display
            enlarged_frame = cv2.resize(frame_rgb, 
                                      (96 * SCALE_FACTOR, 96 * SCALE_FACTOR),
                                      interpolation=cv2.INTER_NEAREST)
            
            ret, buffer = cv2.imencode('.jpg', enlarged_frame)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            time.sleep(0.1)

def run_detections(frame):
    # YOLOv8 Inference
    if model_yolo is not None:
        try:
            results = model_yolo(frame)
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = model_yolo.names.get(int(box[-1]), "Human")
                    conf = float(box[-2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        except Exception as e:
            logger.error(f"YOLO error: {str(e)}")

    # Roboflow Inference
    if model_roboflow is not None:
        try:
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            predictions = model_roboflow.predict(temp_path, confidence=40, overlap=30).json()
            
            if "predictions" in predictions:
                for obj in predictions["predictions"]:
                    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
                    x1, y1 = x - w//2, y - h//2
                    x2, y2 = x + w//2, y + h//2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frame, f"{obj['class']} {obj['confidence']:.2f}",
                              (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        except Exception as e:
            logger.error(f"Roboflow error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global camera_active
    action = request.form.get('action')
    logger.info(f"Received control action: {action}")
    
    if action == 'start':
        camera_active = True
        logger.info("Starting camera...")
    elif action == 'stop':
        camera_active = False
        release_camera()
        logger.info("Camera stopped")
    
    return '', 204

if __name__ == '__main__':
    init_models()  # Initialize models at startup
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
