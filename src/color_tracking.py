import cv2
import numpy as np

from src.config import DEFAULT_TOLERANCE


class ColorTracker:
    def __init__(self, tolerance=DEFAULT_TOLERANCE):
        self.reference_color = None
        self.tolerance = tolerance

    def set_reference_color(self, color):
        self.reference_color = np.asarray(color, dtype=int)

    def clear_reference(self):
        self.reference_color = None

    def update(self, frame, use_contours=False):
        height, width = frame.shape[:2]
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        bgr_str = "None"
        pixel_count = 0
        sum_x = 0
        sum_y = 0
        center_x = None
        center_y = None

        if self.reference_color is not None:
            bgr_str = (
                f"B:{self.reference_color[0]} "
                f"G:{self.reference_color[1]} "
                f"R:{self.reference_color[2]}"
            )
            diff = np.abs(frame.astype(int) - self.reference_color)
            match_mask = (
                (diff[:, :, 0] <= self.tolerance)
                & (diff[:, :, 1] <= self.tolerance)
                & (diff[:, :, 2] <= self.tolerance)
            )
            binary_mask = (match_mask * 255).astype(np.uint8)

            if use_contours:
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    filtered_mask = np.zeros_like(binary_mask)
                    cv2.drawContours(filtered_mask, [largest_contour], -1, 255, thickness=-1)
                    binary_mask = filtered_mask
                else:
                    binary_mask = np.zeros_like(binary_mask)

            ys, xs = np.where(binary_mask > 0)
            pixel_count = len(xs)

            if pixel_count > 0:
                sum_x = int(np.sum(xs))
                sum_y = int(np.sum(ys))
                center_x = int(sum_x / pixel_count)
                center_y = int(sum_y / pixel_count)

        return {
            "bgr_str": bgr_str,
            "binary_mask": binary_mask,
            "pixel_count": pixel_count,
            "sum_x": sum_x,
            "sum_y": sum_y,
            "center_x": center_x,
            "center_y": center_y,
        }
