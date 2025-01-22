from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import json

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Global variables for landmark data
landmark_data = []
lock = threading.Lock()

def process_video():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            continue
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            
            for landmark in face_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            
            with lock:
                global landmark_data
                landmark_data = landmarks
        
        time.sleep(0.03)  # Limit processing rate

@app.route('/landmarks')
def get_landmarks():
    with lock:
        return jsonify(landmark_data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Start video processing in background thread
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
