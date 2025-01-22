import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
import time
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = []
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append([x, y])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                with thread_lock:
                    latest_landmarks = landmarks

        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')

        with thread_lock:
            latest_frame = encoded_frame

        time.sleep(0.033)  # ~30 FPS


@app.route('/api/video')
def video_feed():
    with thread_lock:
        if latest_frame is None:
            return jsonify({'error': 'No frame available'}), 200
        return jsonify({
            'frame': latest_frame,
            'landmarks': latest_landmarks
        }), 200


@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'timestamp': time.time()
    }), 200


@app.route('/')
def home():
    return "Python Backend Server is Running"


if __name__ == '__main__':
    video_thread = threading.Thread(target=process_webcam)
    video_thread.daemon = True
    video_thread.start()
    app.run(host='0.0.0.0', port=5000)
