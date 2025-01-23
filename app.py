import asyncio
import websockets
import cv2
import numpy as np

connected_clients = set()

async def detect_face(websocket, path):
    global connected_clients
    connected_clients.add(websocket)
    try:
        while True:
            message = await websocket.recv()
            if message == 'camera_connected':
                # Kamera sudah terhubung, mulai proses frame dari OpenCV
                print("Camera connected, starting face detection...")
                cap = cv2.VideoCapture(0)  # Mengakses kamera

                if not cap.isOpened():
                    print("Error: Camera not found or cannot be opened.")
                    break

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        break

                    # Deteksi wajah (bisa disesuaikan dengan model deteksi wajah Anda)
                    faces = detect_faces(frame)

                    # Kirim frame ke klien
                    await websocket.send(frame.tobytes())  # Kirim data frame (pastikan formatnya sesuai)
            else:
                print(f"Received message: {message}")
    except websockets.ConnectionClosed:
        print("Connection closed")
    finally:
        connected_clients.remove(websocket)

def detect_faces(frame):
    # Deteksi wajah (ini contoh sederhana)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

async def main():
    async with websockets.serve(detect_face, "localhost", 8089):
        print("WebSocket server started at ws://localhost:8089")
        await asyncio.Future()  # Menunggu selamanya

if __name__ == "__main__":
    asyncio.run(main())
