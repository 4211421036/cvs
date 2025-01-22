from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
import time
import pyngrok.conf
from pyngrok import ngrok
import socket
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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
public_url = None

def get_ip():
    """Get the local IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def setup_ngrok():
    """Setup and start ngrok tunnel"""
    global public_url
    try:
        # Configure ngrok
        ngrok.set_auth_token("2rzrmo6wV412WC59qNhdvlMTspg_6HtjDgvjMA1vGzh55r2ta")  # Replace with your ngrok auth token
        
        # Start ngrok tunnel
        public_url = ngrok.connect(5000).public_url
        logger.info("\n" + "=" * 50)
        logger.info(f"Local IP: http://{get_ip()}:5000")
        logger.info(f"Public URL: {public_url}")
        logger.info("=" * 50 + "\n")
        
        return public_url
    except Exception as e:
        logger.error(f"Error setting up ngrok: {e}")
        return None

def process_webcam():
    """Process webcam feed and detect landmarks"""
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
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                with thread_lock:
                    latest_landmarks = landmarks
        
        # Add URL information to frame
        if public_url:
            cv2.putText(frame, f"Server URL: {public_url}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        with thread_lock:
            latest_frame = encoded_frame
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def home():
    """Home page with server information"""
    return f"""
    <html>
        <head>
            <title>Face Landmark Detection Server</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                }}
                .url-box {{
                    background: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .copy-btn {{
                    padding: 5px 10px;
                    margin-left: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>Face Landmark Detection Server</h1>
            <div class="url-box">
                <strong>Local URL:</strong> 
                <span id="local-url">http://{get_ip()}:5000</span>
                <button class="copy-btn" onclick="copyToClipboard('local-url')">Copy</button>
            </div>
            <div class="url-box">
                <strong>Public URL:</strong> 
                <span id="public-url">{public_url}</span>
                <button class="copy-btn" onclick="copyToClipboard('public-url')">Copy</button>
            </div>
            <p>Use these URLs in your GitHub Pages frontend to connect to this server.</p>
            <script>
                function copyToClipboard(elementId) {{
                    const text = document.getElementById(elementId).textContent;
                    navigator.clipboard.writeText(text);
                    alert('URL copied to clipboard!');
                }}
            </script>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Endpoint for video feed"""
    global latest_frame
    with thread_lock:
        if latest_frame is None:
            return jsonify({'error': 'No frame available'})
        return jsonify({
            'frame': latest_frame,
            'landmarks': latest_landmarks,
            'server_url': public_url
        })

@app.route('/server_info')
def server_info():
    """Endpoint for server information"""
    return jsonify({
        'local_url': f"http://{get_ip()}:5000",
        'public_url': public_url
    })

if __name__ == '__main__':
    # Setup ngrok
    public_url = setup_ngrok()
    
    # Start the webcam processing thread
    video_thread = threading.Thread(target=process_webcam)
    video_thread.daemon = True
    video_thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
