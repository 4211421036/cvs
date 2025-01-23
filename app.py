import asyncio
import websockets
import cv2
import base64
import numpy as np
import io
from aiohttp import web

async def detect_face(websocket, path):
    # Baca frame dari WebSocket
    try:
        while True:
            frame_data = await websocket.recv()
            # Mengkonversi base64 frame menjadi gambar
            img_data = base64.b64decode(frame_data.split(',')[1])  # Hapus bagian 'data:image/jpeg;base64,'
            img = np.array(bytearray(img_data), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # Deteksi wajah di frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Mengirim kembali gambar yang telah diproses sebagai base64
            _, buffer = cv2.imencode('.jpg', img)
            jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(f"data:image/jpeg;base64,{jpg_as_base64}")

    except websockets.ConnectionClosed:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    # Menjalankan server WebSocket di localhost:8089
    server = await websockets.serve(detect_face, "localhost", 8089)
    print("WebSocket server started at ws://localhost:8089")
    await server.wait_closed()

# Jalankan server WebSocket
asyncio.get_event_loop().run_until_complete(main())
