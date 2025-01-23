import asyncio
import websockets
import cv2
import base64
import json
import signal
from camera import Camera
from httpserver import HttpServer

import os
from camera import Camera, MockCamera

# Gunakan MockCamera jika kamera fisik tidak tersedia
if os.environ.get("USE_MOCK_CAMERA", "false").lower() == "true":
    camera = MockCamera(0)
else:
    camera = Camera(0)

# Inisialisasi komponen
camera = Camera(0)
httpserver = HttpServer(8088)
clients = set()

async def handler(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def send(websocket):
    frame = camera.get_frame()
    faces = camera.get_faces()
    ret, encoded = cv2.imencode(".png", frame)
    if ret:
        base64Frame = base64.b64encode(encoded).decode("ascii")
        payload = {"frame": base64Frame, "faces": faces}
        try:
            await websocket.send(json.dumps(payload))
        except (websockets.ConnectionClosed, AssertionError):
            pass
    else:
        print("Failed to encode frame")

async def broadcast():
    while True:
        tasks = [send(client) for client in clients]
        if tasks:
            await asyncio.gather(*tasks)
        await asyncio.sleep(0.04)

async def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Tangani Ctrl+C

    try:
        httpserver.start()
        camera.start()
        async with websockets.serve(handler, "0.0.0.0", 5000):
            await broadcast()
    except (KeyboardInterrupt, asyncio.CancelledError):
        httpserver.stop()
        camera.stop()

if __name__ == "__main__":
    asyncio.run(main())
