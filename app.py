import asyncio
import websockets
import cv2
import base64
from camera import Camera
from httpserver import HttpServer
import signal
import json

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
    if (ret):
        base64Frame = base64.b64encode(encoded).decode("ascii")
        try:
            # await websocket.send(base64Frame)
            payload = {}
            payload['frame'] = base64Frame
            payload['faces'] = faces
            await websocket.send(json.dumps(payload))
        except (websockets.ConnectionClosed, AssertionError):
            pass
    else:
        print("failed to encode frame")


async def broadcast():
    while True:
        for websocket in clients:
            await send(websocket)
        await asyncio.sleep(0.04)


async def main():
    # This restores the default Ctrl+C signal handler, which just kills the process
    # This is a workaround with the problem of asyncio cleaning up the websockets with Ctrl+C
    # Remove this line and it will shutdown gracefully if no client is connected but will not once a connection is made
    # TODO find a better way to gracefully clean-up and shutdown
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
