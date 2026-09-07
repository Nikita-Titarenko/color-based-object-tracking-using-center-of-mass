import cv2

from src.config import haar_cascade_path


class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(haar_cascade_path())

    def detect(self, frame):
        if self.cascade.empty():
            return "Not Found", "N/A", [], []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        candidates = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=0)

        if len(faces) == 0:
            return "Not Found", "N/A", [], candidates

        fx, fy, fw, fh = faces[0]
        return f"Detected ({len(faces)})", f"x:{fx} y:{fy} w:{fw} h:{fh}", faces, candidates
