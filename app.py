from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables for storing the latest frame and landmarks
latest_frame = None
latest_landmarks = []
thread_lock = threading.Lock()

def process_webcam():
    global latest_frame, latest_landmarks
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            continue
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        # Draw landmarks on frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
                    # Draw landmarks
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                with thread_lock:
                    latest_landmarks = landmarks
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        with thread_lock:
            latest_frame = encoded_frame
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    global latest_frame
    with thread_lock:
        if latest_frame is None:
            return jsonify({'error': 'No frame available'})
        return jsonify({
            'frame': latest_frame,
            'landmarks': latest_landmarks
        })

if __name__ == '__main__':
    # Start the webcam processing thread
    video_thread = threading.Thread(target=process_webcam)
    video_thread.daemon = True
    video_thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
