import cv2
import numpy as np

from src.color_tracking import ColorTracker
from src.config import DEFAULT_TOLERANCE, VIDEO_HEIGHT, VIDEO_WIDTH, WINDOW_NAME, get_screen_size
from src.dashboard import render_dashboard
from src.face_detection import FaceDetector
from src.video_source import choose_video_file, open_video_source


class App:
    def __init__(self):
        self.tracker = ColorTracker(DEFAULT_TOLERANCE)
        self.face_detector = FaceDetector()
        self.capture = None
        self.current_frame = self._create_placeholder_frame()
        self.last_frame = None
        self.source_type = "idle"
        self.video_path = None
        self.play_state = "idle"

    def _create_placeholder_frame(self):
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Select a source",
            (160, 220),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _open_capture(self, source_type, video_path=None):
        if source_type == "video":
            return open_video_source(video_path)
        return open_video_source()

    def set_source(self, source_type, video_path=None):
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

        self.source_type = source_type
        self.video_path = video_path
        self.capture = self._open_capture(source_type, video_path)
        self.play_state = "playing" if self.capture.isOpened() else "idle"

    def _detect_button(self, x, y):
        if self.play_state == "idle":
            buttons = {
                "choose_video": (30, 30, 220, 48),
                "use_camera": (30, 95, 220, 48),
            }
        elif self.play_state == "ended":
            buttons = {
                "restart_video": (30, 30, 240, 48),
                "choose_video": (30, 95, 240, 48),
                "use_camera": (30, 160, 240, 48),
            }
        else:
            return None

        for action, rect in buttons.items():
            bx, by, bw, bh = rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return action
        return None

    def on_mouse_event(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x < VIDEO_WIDTH and y < VIDEO_HEIGHT:
            action = self._detect_button(x, y)
            if action == "choose_video":
                chosen = choose_video_file()
                if chosen:
                    self.set_source("video", chosen)
                return
            if action == "use_camera":
                self.set_source("camera")
                return
            if action == "restart_video":
                if self.video_path:
                    self.set_source("video", self.video_path)
                return

        frame = self.current_frame
        if frame is not None and self.play_state == "playing" and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            self.tracker.set_reference_color(frame[y, x].astype(int))

    def run(self):
        screen_width, screen_height = get_screen_size()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, screen_width, screen_height)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse_event, self)

        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if self.capture is not None and self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    self.play_state = "ended"
                    frame = self.last_frame if self.last_frame is not None else self._create_placeholder_frame()
                else:
                    self.last_frame = frame.copy()
                    self.play_state = "playing"
            else:
                frame = self.last_frame if self.last_frame is not None else self._create_placeholder_frame()
                if self.play_state == "idle":
                    frame = self._create_placeholder_frame()

            self.current_frame = frame.copy()
            color_info = self.tracker.update(frame)
            face_status, face_coords, _ = self.face_detector.detect(frame)
            canvas = render_dashboard(frame, color_info, face_status, face_coords, self.tracker.tolerance, {"play_state": self.play_state})
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                self.tracker.clear_reference()
            if self.play_state == "ended" and key == ord("r") and self.video_path:
                self.set_source("video", self.video_path)

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()


def main():
    app = App()
    app.run()
