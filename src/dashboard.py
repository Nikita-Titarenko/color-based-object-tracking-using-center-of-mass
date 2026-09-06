import cv2
import numpy as np

from src.config import PANEL_WIDTH, WINDOW_NAME, get_screen_size


def draw_panel_text(panel, x, y, text, font_scale=0.85, color=(255, 255, 255), thickness=1):
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return int(y + max(28, 22 * font_scale + 8))


def draw_button(panel, x, y, w, h, label, fill_color=(31, 120, 255), border_color=(255, 255, 255), text_color=(255, 255, 255)):
    cv2.rectangle(panel, (x, y), (x + w, y + h), border_color, 2)
    cv2.rectangle(panel, (x, y), (x + w, y + h), fill_color, -1)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    tx = x + (w - text_size[0]) // 2
    ty = y + (h + text_size[1]) // 2
    cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1, cv2.LINE_AA)
    return (x, y, w, h)


def get_button_rects(state):
    if state["play_state"] == "idle":
        return {
            "choose_video": (30, 30, 220, 48),
            "use_camera": (30, 95, 220, 48),
        }
    if state["play_state"] == "ended":
        return {
            "restart_video": (30, 30, 240, 48),
            "choose_video": (30, 95, 240, 48),
            "use_camera": (30, 160, 240, 48),
        }
    return {}


def draw_buttons(video_panel, state):
    for name, rect in get_button_rects(state).items():
        x, y, w, h = rect
        label = {
            "choose_video": "Choose Video",
            "use_camera": "Use Camera",
            "restart_video": "Restart Video",
        }[name]
        draw_button(video_panel, x, y, w, h, label)


def render_dashboard(frame, color_info, face_status, face_coords, tolerance, state):
    if frame is None:
        height, width = 480, 640
        display_frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        height, width, _ = frame.shape
        display_frame = frame.copy()

    screen_width, screen_height = get_screen_size()
    target_width = max(640, int(screen_width * 0.64))
    target_height = max(480, int(screen_height * 0.75))
    if display_frame.shape[1] != target_width or display_frame.shape[0] != target_height:
        display_frame = cv2.resize(display_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        height, width = display_frame.shape[:2]

    center_x = color_info["center_x"]
    center_y = color_info["center_y"]
    binary_mask = color_info["binary_mask"]

    if center_x is not None and center_y is not None:
        cv2.circle(display_frame, (center_x, center_y), 7, (0, 0, 255), -1)
        cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
        cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)

    if face_status.startswith("Detected"):
        parts = face_coords.replace("x:", " ").replace("y:", " ").replace("w:", " ").replace("h:", " ").split()
        if len(parts) == 4:
            x, y, w, h = map(int, parts)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if state["play_state"] == "idle":
        cv2.putText(display_frame, "Select a video source", (80, 220), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    elif state["play_state"] == "ended":
        cv2.putText(display_frame, "Video finished", (150, 220), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    draw_buttons(display_frame, state)

    canvas = np.zeros((height, width + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :width] = display_frame

    panel = canvas[:, width:]
    panel[:] = (30, 30, 30)
    y_offset = 25

    y_offset = draw_panel_text(panel, 10, y_offset, "=== COLOR TRACKING (LK 1) ===", 1.0, (0, 255, 255), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, f"Formula: |C_frame - C_ref| <= {tolerance}", 0.72, (200, 200, 200), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, f"Target Ref Color: {color_info['bgr_str']}", 0.72, (255, 255, 255), 1)
    y_offset += 16

    mask_h, mask_w = 150, 200
    resized_mask = cv2.resize(binary_mask, (mask_w, mask_h))
    mask_3ch = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

    y_offset = draw_panel_text(panel, 10, y_offset, "Current Binary Mask:", 0.72, (200, 200, 200), 1)
    y_offset += 8
    panel[y_offset:y_offset + mask_h, 10:10 + mask_w] = mask_3ch
    cv2.rectangle(panel, (10, y_offset), (10 + mask_w, y_offset + mask_h), (0, 0, 255), 2)
    y_offset += mask_h + 20

    y_offset = draw_panel_text(panel, 10, y_offset, "=== CENTER OF MASS (Xc, Yc) ===", 1.0, (0, 255, 255), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, f"N (pixel count): {color_info['pixel_count']}", 0.78, (255, 255, 255), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, f"Sum X (sum x_i): {color_info['sum_x']}", 0.78, (200, 200, 200), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, f"Sum Y (sum y_i): {color_info['sum_y']}", 0.78, (200, 200, 200), 1)

    center_str = f"({center_x}, {center_y})" if center_x is not None else "None"
    y_offset = draw_panel_text(panel, 10, y_offset, f"Result (Xc, Yc): {center_str}", 0.85, (0, 0, 255), 2)
    y_offset += 20

    y_offset = draw_panel_text(panel, 10, y_offset, "=== HAAR CASCADE (LR 3) ===", 1.0, (0, 255, 0), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, f"Face Status: {face_status}", 0.78, (255, 255, 255), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, f"Coords (x,y,w,h): {face_coords}", 0.72, (200, 200, 200), 1)
    y_offset += 12

    y_offset = draw_panel_text(panel, 10, y_offset, "=== COMPARISON SUMMARY ===", 0.8, (255, 255, 255), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, "LK1: Fast pixel search, color-sensitive", 0.65, (180, 180, 180), 1)
    y_offset = draw_panel_text(panel, 10, y_offset, "LR3: Feature-based, robust face detect", 0.65, (180, 180, 180), 1)

    return canvas


def show_dashboard(canvas):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, canvas.shape[1], canvas.shape[0])
    cv2.imshow(WINDOW_NAME, canvas)
