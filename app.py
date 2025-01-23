import asyncio
import websockets
import cv2
import mediapipe as mp
import base64
import json

class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture frame")
        return frame

    def get_faces(self):
        frame = self.get_frame()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face = [
                    (lm.x, lm.y, lm.z) for lm in face_landmarks.landmark
                ]
                landmarks.append(face)
        return landmarks

    def stop(self):
        self.cap.release()


async def handler(websocket):
    camera = Camera(0)
    try:
        while True:
            frame = camera.get_frame()
            faces = camera.get_faces()

            ret, buffer = cv2.imencode(".png", frame)
            base64_frame = base64.b64encode(buffer).decode("ascii")

            payload = {
                "frame": base64_frame,
                "faces": faces
            }
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.04)  # ~25 FPS
    finally:
        camera.stop()


async def main():
    async with websockets.serve(handler, "localhost", 8089):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
