# adapted from https://nickhuber.ca/blog/python-opencv-camera-websockets

import threading
import time
import cv2
import json


class Camera:

    def __init__(self, camIndex):
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.thread = None
        self.current_frame = None
        self.is_running: bool = False
        self.camera = cv2.VideoCapture(camIndex)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")
        if self.camera.isOpened():
            # obtain the dimensions of the video
            self.frame_width = self.camera.get(
                cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            self.frame_height = self.camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    def __del__(self):
        self.camera.release()

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._capture)
            self.thread.start()

    def get_frame(self):
        return self.current_frame

    def get_faces(self):
        return self.faces

    def stop(self):
        self.is_running = False
        self.thread.join()
        self.thread = None
        self.current_frame = None

    def _capture(self):
        self.is_running = True
        while self.is_running:
            time.sleep(0.04)
            ret, frame = self.camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                abs_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                rel_faces = []
                index = 0
                for (x, y, w, h) in abs_faces:
                    index += 1
                    face = {}
                    face['id'] = index
                    face['x'] = x / self.frame_width
                    face['y'] = y / self.frame_height
                    face['w'] = w / self.frame_width
                    face['h'] = h / self.frame_height
                    rel_faces.append(face)
                self.current_frame = frame
                self.faces = rel_faces
            else:
                print("Failed to capture frame")

        print("Camera thread stopped")
        self.thread = None
        self.is_running = False
        self.current_frame = None
