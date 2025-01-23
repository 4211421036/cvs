import asyncio
import websockets
import cv2
import base64
from camera import Camera, MockCamera  # Import MockCamera
from httpserver import HttpServer
import signal
import json
import os

# Use MockCamera in CI/CD environment
if "GITHUB_ACTIONS" in os.environ:
    camera = MockCamera(0)
else:
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
        try:
            payload = {}
            payload['frame'] = base64Frame
            payload['faces'] = faces
            await websocket.send(json.dumps(payload))
        except (websockets.ConnectionClosed, AssertionError):
            pass
    else:
        print("Failed to encode frame")


async def broadcast():
    while True:
        for websocket in clients:
            await send(websocket)
        await asyncio.sleep(0.04)


async def main():
    # Graceful shutdown setup
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        httpserver.start()
        camera.start()
        async with websockets.serve(handler, "", 8089):
            await broadcast()
    except (KeyboardInterrupt, asyncio.CancelledError):
        httpserver.stop()
        camera.stop()


if __name__ == "__main__":
    asyncio.run(main())
