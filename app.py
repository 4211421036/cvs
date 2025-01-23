import cv2
import numpy as np
import base64
import asyncio
import websockets
import json
from flask import Flask, render_template

app = Flask(__name__)

async def detect_face(websocket, path):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        try:
            # Receive image from client
            image_data = await websocket.recv()
            img_data = base64.b64decode(image_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangle around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Convert frame to base64 for sending to the client
            _, buffer = cv2.imencode('.png', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Send the frame with face annotations back to the client
            await websocket.send(json.dumps({'frame': frame_b64}))

        except Exception as e:
            print(f"Error: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Start Flask app and WebSocket server
    loop = asyncio.get_event_loop()
    loop.run_until_complete(websockets.serve(detect_face, 'localhost', 8089))
    app.run(host='0.0.0.0', port=8080)
