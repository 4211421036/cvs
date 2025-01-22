from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
import time
from pyngrok import ngrok
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi ngrok
ngrok.set_auth_token("cr_2rzrmo6wV412WC59qNhdvlMTspg")
public_url = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables
latest_frame = None
latest_landmarks = []
thread_lock = threading.Lock()

def setup_ngrok():
    """Setup dan start ngrok tunnel"""
    global public_url
    try:
        # Start ngrok tunnel
        tunnel = ngrok.connect(5000)
        public_url = tunnel.public_url
        logger.info("\n" + "=" * 50)
        logger.info(f"ngrok tunnel active at: {public_url}")
        logger.info("=" * 50 + "\n")
        return public_url
    except Exception as e:
        logger.error(f"Error setting up ngrok: {e}")
        return None

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
                for landmark in face_landmarks.landmark:
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append([x, y])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                with thread_lock:
                    latest_landmarks = landmarks
        
        # Add URL overlay to frame
        if public_url:
            cv2.putText(frame, f"Server URL: {public_url}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        with thread_lock:
            latest_frame = encoded_frame
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/api/video')
def video_feed():
    global latest_frame, latest_landmarks
    with thread_lock:
        if latest_frame is None:
            return jsonify({'error': 'No frame available'}), 200
        return jsonify({
            'frame': latest_frame,
            'landmarks': latest_landmarks,
            'server_url': public_url
        }), 200

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'server_url': public_url,
        'timestamp': time.time()
    }), 200

@app.route('/')
def home():
    return jsonify({
        'status': 'Server is running',
        'server_url': public_url
    }), 200

if __name__ == '__main__':
    try:
        # Setup ngrok
        logger.info("Starting ngrok tunnel...")
        public_url = setup_ngrok()
        
        if not public_url:
            logger.error("Failed to establish ngrok tunnel")
            exit(1)
            
        # Start video thread
        logger.info("Starting video processing thread...")
        video_thread = threading.Thread(target=process_webcam)
        video_thread.daemon = True
        video_thread.start()
        
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        if 'tunnel' in locals():
            ngrok.disconnect(tunnel.public_url)
