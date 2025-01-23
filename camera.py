import cv2
import numpy as np


class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

    def start(self):
        pass

    def stop(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture frame")
        return frame

    def get_faces(self):
        # Replace with actual face detection logic if needed
        return []


class MockCamera:
    def __init__(self, index):
        # Mock initialization
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self):
        # Return a blank image (black frame)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_faces(self):
        # Return empty face data
        return []
