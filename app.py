from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow
import os
import time

app = Flask(__name__)

# Global variables
camera_active = False
cap = None
model_roboflow = None
model_yolo = None

def init_models():
    global model_roboflow, model_yolo
    try:
        # Initialize Roboflow model
        rf = Roboflow(api_key="vuAfslycGYT8XKOu4lId")
        project = rf.workspace().project("thermal-person-df3lf")
        model_roboflow = project.version(1).model
        print("Roboflow model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Roboflow model: {str(e)}")
        model_roboflow = None

    try:
        # Initialize YOLO model
        model_yolo = YOLO("yolov8n.pt")
        print("YOLO model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize YOLO model: {str(e)}")
        model_yolo = None

def init_camera():
    global cap
    # Try different camera indices
    for camera_index in [ 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Camera opened at index {camera_index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 96)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
            return True
    print("Error: Could not open any camera")
    return False

def release_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None

def generate_frames():
    global camera_active, cap, model_roboflow, model_yolo
    SCALE_FACTOR = 6
    placeholder = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Camera Off", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    while True:
        if not camera_active:
            # Show placeholder when camera is off
            enlarged_placeholder = cv2.resize(placeholder, 
                                           (96 * SCALE_FACTOR, 96 * SCALE_FACTOR),
                                           interpolation=cv2.INTER_NEAREST)
            ret, buffer = cv2.imencode('.jpg', enlarged_placeholder)
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
            time.sleep(0.1)
            continue

        # Process frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.applyColorMap(frame_gray, cv2.COLORMAP_JET)

        # YOLOv8 Inference (if model loaded)
        if model_yolo is not None:
            try:
                results = model_yolo(frame_rgb)
                for result in results:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box[:4])
                        class_id = int(box[-1])
                        label = model_yolo.names.get(class_id, "Human") 
                        conf = box[-2]
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(frame_rgb, f"{label} {conf:.2f}", (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            except Exception as e:
                print(f"YOLO inference error: {str(e)}")

        # Roboflow Inference (if model loaded)
        if model_roboflow is not None:
            temp_image_path = "temp_frame.jpg"
            cv2.imwrite(temp_image_path, frame_rgb)
            try:
                predictions = model_roboflow.predict(temp_image_path, confidence=40, overlap=30).json()
                if "predictions" in predictions:
                    for obj in predictions["predictions"]:
                        x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
                        x1, y1, x2, y2 = x - w//2, y - h//2, x + w//2, y + h//2
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        cv2.putText(frame_rgb, f"{obj['class']} {obj['confidence']:.2f}", 
                                  (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            except Exception as e:
                print(f"Roboflow inference error: {str(e)}")
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

        # Resize for display
        enlarged_frame = cv2.resize(frame_rgb, 
                                 (96 * SCALE_FACTOR, 96 * SCALE_FACTOR),
                                 interpolation=cv2.INTER_NEAREST)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', enlarged_frame)
        if ret:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
    
    if action == 'start':
        camera_active = True
    elif action == 'stop':
        camera_active = False
        release_camera()
    
    return '', 204

if __name__ == '__main__':
    init_models()  # Initialize models when starting the app
    app.run(debug=True)