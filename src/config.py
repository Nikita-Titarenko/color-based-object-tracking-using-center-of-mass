import cv2

WINDOW_NAME = "Color-based Centroid Tracking and Haar Cascade Classifier"
PANEL_WIDTH = 650
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
DEFAULT_TOLERANCE = 10
HAAR_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
VIDEO_SELECTION_TITLE = "Select a video file for analysis (or cancel to use the webcam)"
VIDEO_FILE_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
    ("All files", "*.*"),
]
WINDOW_SIZE = (1280, 720)


def get_screen_size():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


def haar_cascade_path():
    return cv2.data.haarcascades + HAAR_CASCADE_FILENAME
