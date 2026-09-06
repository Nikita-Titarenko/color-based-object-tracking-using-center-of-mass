import cv2
import tkinter as tk
from tkinter import filedialog

from src.config import VIDEO_FILE_TYPES, VIDEO_SELECTION_TITLE


def choose_video_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title=VIDEO_SELECTION_TITLE,
        filetypes=VIDEO_FILE_TYPES,
    )
    root.destroy()
    return file_path


def open_video_source(video_path=None):
    if video_path:
        return cv2.VideoCapture(video_path)
    return cv2.VideoCapture(0)
