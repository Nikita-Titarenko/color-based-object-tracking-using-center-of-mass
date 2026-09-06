import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from src.color_tracking import ColorTracker
from src.config import DEFAULT_TOLERANCE, WINDOW_NAME
from src.face_detection import FaceDetector
from src.video_source import choose_video_file, open_video_source


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_NAME)
        self.root.configure(bg="#10151d")
        self.root.geometry("1600x900")
        self.root.state("zoomed")
        self.root.minsize(1200, 700)

        self.tracker = ColorTracker(DEFAULT_TOLERANCE)
        self.face_detector = FaceDetector()
        self.capture = None
        self.frame = self._create_placeholder_frame()
        self.last_frame = None
        self.video_path = None
        self.source_type = None
        self.play_state = "idle"
        self.mask_probe_point = None

        self._build_ui()
        self._update_dashboard()

    def _create_placeholder_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    def _get_video_display_geometry(self, src_width, src_height):
        target_width = self.video_area.winfo_width() or 600
        target_height = self.video_area.winfo_height() or 400

        scale = min(target_width / src_width, target_height / src_height)
        display_width = max(1, int(src_width * scale))
        display_height = max(1, int(src_height * scale))
        offset_x = (target_width - display_width) // 2
        offset_y = (target_height - display_height) // 2

        return target_width, target_height, display_width, display_height, offset_x, offset_y

    def _get_mask_preview_geometry(self, src_width, src_height):
        target_width = self.mask_preview.winfo_width() or 380
        target_height = self.mask_preview.winfo_height() or 220

        scale = min(target_width / src_width, target_height / src_height)
        display_width = max(1, int(src_width * scale))
        display_height = max(1, int(src_height * scale))
        offset_x = (target_width - display_width) // 2
        offset_y = (target_height - display_height) // 2

        return target_width, target_height, display_width, display_height, offset_x, offset_y

    def _fit_frame_to_video_area(self, image):
        src_width, src_height = image.size
        target_width, target_height, resized_width, resized_height, offset_x, offset_y = self._get_video_display_geometry(src_width, src_height)

        resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        canvas.paste(resized_image, (offset_x, offset_y))
        return canvas

    def _fit_mask_to_preview(self, mask):
        src_height, src_width = mask.shape[:2]
        target_width, target_height, resized_width, resized_height, offset_x, offset_y = self._get_mask_preview_geometry(src_width, src_height)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        resized_mask = cv2.resize(mask_rgb, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized_mask
        return Image.fromarray(canvas)

    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=5)
        self.root.grid_columnconfigure(1, weight=4, minsize=560)

        left_panel = tk.Frame(self.root, bg="#0d1117")
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=0)
        left_panel.grid_columnconfigure(0, weight=1)

        self.video_area = tk.Label(
            left_panel,
            bg="#000000",
            highlightthickness=0,
            anchor="center",
            justify="center",
        )
        self.video_area.grid(row=0, column=0, sticky="nsew", padx=18, pady=(18, 10))
        self.video_area.bind("<Button-1>", self._on_video_click)

        controls_frame = tk.Frame(left_panel, bg="#0d1117")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 18))
        controls_frame.grid_columnconfigure(0, weight=1)

        button_style = {
            "bg": "#ff7a00",
            "fg": "white",
            "activebackground": "#ff8f1a",
            "activeforeground": "white",
            "font": ("Segoe UI", 18, "bold"),
            "bd": 0,
            "cursor": "hand2",
            "relief": "flat",
            "padx": 14,
            "pady": 12,
            "highlightthickness": 0,
        }

        self.choose_video_btn = tk.Button(controls_frame, text="Choose Video", command=self._choose_video, **button_style)
        self.choose_video_btn.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.use_camera_btn = tk.Button(controls_frame, text="Use Camera", command=self._use_camera, **button_style)
        self.use_camera_btn.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self.stop_resume_btn = tk.Button(controls_frame, text="Stop", command=self._toggle_pause_resume, state="disabled", **button_style)
        self.stop_resume_btn.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.retry_btn = tk.Button(controls_frame, text="Retry", command=self._restart_video, state="normal", **button_style)
        self.retry_btn.grid(row=3, column=0, sticky="ew")

        right_panel_container = tk.Frame(self.root, bg="#1a1f2b")
        right_panel_container.grid(row=0, column=1, sticky="nsew")

        sidebar_canvas = tk.Canvas(
            right_panel_container,
            bg="#1a1f2b",
            highlightthickness=0,
            bd=0,
        )
        sidebar_scrollbar = tk.Scrollbar(right_panel_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar_scrollbar.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)

        right_panel = tk.Frame(sidebar_canvas, bg="#1a1f2b")
        sidebar_window = sidebar_canvas.create_window((0, 0), window=right_panel, anchor="nw")

        def _update_sidebar_scrollregion(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def _resize_sidebar_content(event):
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)

        def _on_sidebar_mousewheel(event):
            hovered = self.root.winfo_containing(event.x_root, event.y_root)
            if hovered is None:
                return
            if str(hovered).startswith(str(right_panel_container)):
                sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        right_panel.bind("<Configure>", _update_sidebar_scrollregion)
        sidebar_canvas.bind("<Configure>", _resize_sidebar_content)
        self.root.bind_all("<MouseWheel>", _on_sidebar_mousewheel, add="+")

        self.header = tk.Label(
            right_panel,
            text="=== COLOR TRACKING ===",
            bg="#1a1f2b",
            fg="#ffd000",
            font=("Consolas", 13, "bold"),
            anchor="w",
        )
        self.header.pack(fill="x", padx=24, pady=(15, 6))

        self.tolerance_input_var = tk.StringVar(value=str(self.tracker.tolerance))
        self.target_var = tk.StringVar(value="Target Ref Color: None")
        self.mask_var = tk.StringVar(value="Current Binary Mask:")
        self.use_contours_var = tk.BooleanVar(value=False)
        self.center_title_var = tk.StringVar(value="=== CENTER OF MASS (Xc, Yc) ===")
        self.n_var = tk.StringVar(value="N (pixel count): 0")
        self.sum_x_var = tk.StringVar(value="Sum X (sum x_i): 0")
        self.sum_y_var = tk.StringVar(value="Sum Y (sum y_i): 0")
        self.result_var = tk.StringVar(value="Result (Xc, Yc): None")
        self.face_title_var = tk.StringVar(value="=== HAAR CASCADE ===")
        self.face_status_var = tk.StringVar(value="Face Status: Not Found")
        self.face_coords_var = tk.StringVar(value="Coords (x,y,w,h): N/A")
        self.mask_probe_title_var = tk.StringVar(value="Mask Probe: click binary matrix")
        self.mask_probe_line1_var = tk.StringVar(value="Point: N/A")
        self.mask_probe_diff_b_var = tk.StringVar(value="Diff B: N/A")
        self.mask_probe_diff_g_var = tk.StringVar(value="Diff G: N/A")
        self.mask_probe_diff_r_var = tk.StringVar(value="Diff R: N/A")
        self.mask_probe_check_var = tk.StringVar(value="")
        self.mask_probe_inclusion_var = tk.StringVar(value="Inclusion: N/A")

        text_style = {"bg": "#1a1f2b", "fg": "white", "font": ("Consolas", 12), "anchor": "w"}
        entry_style = {
            "bg": "#0d1117",
            "fg": "white",
            "insertbackground": "white",
            "font": ("Consolas", 12),
            "relief": "solid",
            "borderwidth": 1,
        }

        tolerance_row = tk.Frame(right_panel, bg="#1a1f2b")
        tolerance_row.pack(anchor="w", fill="x", padx=24, pady=(6, 2))

        tk.Label(tolerance_row, text="Tolerance:", **text_style).pack(side="left")
        tolerance_entry = tk.Entry(tolerance_row, textvariable=self.tolerance_input_var, width=6, **entry_style)
        tolerance_entry.pack(side="left", padx=(10, 0))
        tolerance_entry.bind("<Return>", self._on_tolerance_submit)
        tk.Button(
            tolerance_row,
            text="Set",
            command=self._on_tolerance_submit,
            bg="#ff7a00",
            fg="white",
            activebackground="#ff8f1a",
            activeforeground="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief="flat",
            padx=10,
            pady=4,
            highlightthickness=0,
            cursor="hand2",
        ).pack(side="left", padx=(10, 0))

        target_row = tk.Frame(right_panel, bg="#1a1f2b")
        target_row.pack(anchor="w", fill="x", padx=24, pady=(2, 0))

        tk.Label(target_row, textvariable=self.target_var, **text_style).pack(side="left")
        self.target_color_preview = tk.Label(
            target_row,
            width=2,
            height=1,
            bg="#2b3140",
            relief="solid",
            borderwidth=1,
        )
        self.target_color_preview.pack(side="left", padx=(10, 0), ipady=6)

        tk.Checkbutton(
            right_panel,
            text="Use findContours filter",
            variable=self.use_contours_var,
            onvalue=True,
            offvalue=False,
            bg="#1a1f2b",
            fg="#ffffff",
            activebackground="#1a1f2b",
            activeforeground="#ffffff",
            selectcolor="#2b3140",
            font=("Consolas", 12),
            anchor="w",
            padx=4,
        ).pack(anchor="w", padx=24, pady=(4, 2))
        tk.Label(right_panel, textvariable=self.mask_var, **text_style).pack(anchor="w", padx=24, pady=(10, 6))

        mask_panel = tk.Frame(right_panel, bg="#1a1f2b")
        mask_panel.pack(anchor="w", padx=24, pady=(0, 10))

        self.mask_preview = tk.Label(mask_panel, bg="#000000", width=380, height=220, relief="solid", borderwidth=2)
        self.mask_preview.pack_propagate(False)
        self.mask_preview.pack(anchor="w", padx=0, pady=(0, 10))
        self.mask_preview.bind("<Button-1>", self._on_mask_preview_click)

        tk.Label(mask_panel, textvariable=self.mask_probe_title_var, bg="#1a1f2b", fg="#ffd000", font=("Consolas", 13, "bold"), anchor="w").pack(anchor="w", padx=0, pady=(0, 2))
        probe_color_row = tk.Frame(mask_panel, bg="#1a1f2b")
        probe_color_row.pack(anchor="w", fill="x", padx=0, pady=(0, 2))
        tk.Label(probe_color_row, text="Pixel color:", bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w").pack(side="left")
        self.mask_probe_color_preview = tk.Label(
            probe_color_row,
            width=2,
            height=1,
            bg="#2b3140",
            relief="solid",
            borderwidth=1,
        )
        self.mask_probe_color_preview.pack(side="left", padx=(10, 0), ipady=5)
        tk.Label(mask_panel, textvariable=self.mask_probe_line1_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", fill="x", padx=0)
        tk.Label(mask_panel, textvariable=self.mask_probe_diff_b_var, bg="#1a1f2b", fg="#d0d0d0", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", fill="x", padx=0)
        tk.Label(mask_panel, textvariable=self.mask_probe_diff_g_var, bg="#1a1f2b", fg="#d0d0d0", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", fill="x", padx=0)
        tk.Label(mask_panel, textvariable=self.mask_probe_diff_r_var, bg="#1a1f2b", fg="#d0d0d0", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", fill="x", padx=0)
        tk.Label(mask_panel, textvariable=self.mask_probe_check_var, bg="#1a1f2b", fg="#d0d0d0", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", fill="x", padx=0, pady=(0, 6))
        tk.Label(mask_panel, textvariable=self.mask_probe_inclusion_var, bg="#1a1f2b", fg="#d0d0d0", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", fill="x", padx=0, pady=(0, 6))

        tk.Label(right_panel, textvariable=self.center_title_var, bg="#1a1f2b", fg="#ffd000", font=("Consolas", 13, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(10, 4))
        tk.Label(right_panel, textvariable=self.n_var, **text_style).pack(anchor="w", padx=24)
        tk.Label(right_panel, textvariable=self.sum_x_var, **text_style).pack(anchor="w", padx=24)
        tk.Label(right_panel, textvariable=self.sum_y_var, **text_style).pack(anchor="w", padx=24)
        tk.Label(right_panel, textvariable=self.result_var, bg="#1a1f2b", fg="#ff0000", font=("Consolas", 12, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(4, 12))

        tk.Label(right_panel, textvariable=self.face_title_var, bg="#1a1f2b", fg="#00ff00", font=("Consolas", 13, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(4, 4))
        tk.Label(right_panel, textvariable=self.face_status_var, bg="#1a1f2b", fg="#00ff00", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24)
        tk.Label(right_panel, textvariable=self.face_coords_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24, pady=(0, 12))

        self.root.bind("<Escape>", lambda event: self.root.destroy())

    def _update_controls_state(self):
        self.retry_btn.config(state="normal")

        can_toggle_playback = self.play_state in ("playing", "paused")
        self.stop_resume_btn.config(
            state="normal" if can_toggle_playback else "disabled",
            text="Resume" if self.play_state == "paused" else "Stop",
        )

    def _on_tolerance_submit(self, event=None):
        try:
            tolerance = int(self.tolerance_input_var.get().strip())
        except ValueError:
            self.tolerance_input_var.set(str(self.tracker.tolerance))
            return

        tolerance = max(0, tolerance)
        self.tracker.tolerance = tolerance
        self.tolerance_input_var.set(str(tolerance))

    def _on_video_click(self, event):
        if self.play_state not in ("playing", "paused"):
            return
        if self.frame is None or self.frame.size == 0:
            return

        frame_h, frame_w = self.frame.shape[:2]
        _, _, display_width, display_height, offset_x, offset_y = self._get_video_display_geometry(frame_w, frame_h)

        if event.x < offset_x or event.x >= offset_x + display_width:
            return
        if event.y < offset_y or event.y >= offset_y + display_height:
            return

        relative_x = (event.x - offset_x) / display_width
        relative_y = (event.y - offset_y) / display_height
        x = max(0, min(int(relative_x * frame_w), frame_w - 1))
        y = max(0, min(int(relative_y * frame_h), frame_h - 1))
        self.tracker.set_reference_color(self.frame[y, x].astype(int))

    def _on_mask_preview_click(self, event):
        if self.frame is None or self.frame.size == 0:
            return

        frame_h, frame_w = self.frame.shape[:2]
        _, _, display_width, display_height, offset_x, offset_y = self._get_mask_preview_geometry(frame_w, frame_h)

        if event.x < offset_x or event.x >= offset_x + display_width:
            return
        if event.y < offset_y or event.y >= offset_y + display_height:
            return

        relative_x = (event.x - offset_x) / display_width
        relative_y = (event.y - offset_y) / display_height
        x = max(0, min(int(relative_x * frame_w), frame_w - 1))
        y = max(0, min(int(relative_y * frame_h), frame_h - 1))
        self.mask_probe_point = (x, y)

    def _update_mask_probe(self, color_info):
        if self.tracker.reference_color is None:
            self.mask_probe_line1_var.set("Point: N/A")
            self.mask_probe_diff_b_var.set("Diff B: choose target color first")
            self.mask_probe_diff_g_var.set("Diff G: choose target color first")
            self.mask_probe_diff_r_var.set("Diff R: choose target color first")
            self.mask_probe_check_var.set("")
            self.mask_probe_inclusion_var.set("Inclusion: N/A")
            self.mask_probe_color_preview.config(bg="#2b3140")
            return

        if self.mask_probe_point is None:
            self.mask_probe_line1_var.set("Point: N/A")
            self.mask_probe_diff_b_var.set("Diff B: click on binary matrix")
            self.mask_probe_diff_g_var.set("Diff G: click on binary matrix")
            self.mask_probe_diff_r_var.set("Diff R: click on binary matrix")
            self.mask_probe_check_var.set("")
            self.mask_probe_inclusion_var.set("Inclusion: N/A")
            self.mask_probe_color_preview.config(bg="#2b3140")
            return

        mask = color_info["binary_mask"]
        mask_h, mask_w = mask.shape[:2]
        x = max(0, min(int(self.mask_probe_point[0]), mask_w - 1))
        y = max(0, min(int(self.mask_probe_point[1]), mask_h - 1))

        blue, green, red = map(int, self.frame[y, x])
        self.mask_probe_color_preview.config(bg=f"#{red:02x}{green:02x}{blue:02x}")
        ref_blue, ref_green, ref_red = map(int, self.tracker.reference_color)
        tol = int(self.tracker.tolerance)

        db = abs(blue - ref_blue)
        dg = abs(green - ref_green)
        dr = abs(red - ref_red)
        pass_b = db <= tol
        pass_g = dg <= tol
        pass_r = dr <= tol
        raw_pass = pass_b and pass_g and pass_r
        mask_pass = int(mask[y, x]) > 0

        self.mask_probe_line1_var.set(
            f"Point: ({x}, {y}) Px BGR=({blue},{green},{red}) Ref=({ref_blue},{ref_green},{ref_red})"
        )
        self.mask_probe_diff_b_var.set(f"Diff B: |{blue}-{ref_blue}|={db} <= {tol} -> {pass_b}")
        self.mask_probe_diff_g_var.set(f"Diff G: |{green}-{ref_green}|={dg} <= {tol} -> {pass_g}")
        self.mask_probe_diff_r_var.set(f"Diff R: |{red}-{ref_red}|={dr} <= {tol} -> {pass_r}")

        if self.use_contours_var.get():
            self.mask_probe_check_var.set(
                f"Threshold check: {'Passed' if raw_pass else 'Failed'}"
            )
        else:
            self.mask_probe_check_var.set("")

        if mask_pass:
            self.mask_probe_inclusion_var.set("Pixel is included in center of mass")
        else:
            self.mask_probe_inclusion_var.set("Pixel is not included in center of mass")

    def _choose_video(self):
        chosen = choose_video_file()
        if not chosen:
            return
        self._set_source("video", chosen)

    def _use_camera(self):
        self._set_source("camera")

    def _restart_video(self):
        if self.capture is not None and self.capture.isOpened() and self.source_type == "video":
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.last_frame = None
            self.play_state = "playing"
            self._update_controls_state()
            return

        if self.video_path:
            self._set_source("video", self.video_path)

    def _toggle_pause_resume(self):
        if self.play_state == "playing":
            self.play_state = "paused"
        elif self.play_state == "paused":
            self.play_state = "playing"
        self._update_controls_state()

    def _set_source(self, source_type, video_path=None):
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

        self.source_type = source_type
        self.video_path = video_path
        if source_type == "video":
            self.capture = open_video_source(video_path)
        else:
            self.capture = open_video_source()

        if self.capture.isOpened():
            self.play_state = "playing"
        else:
            self.play_state = "idle"

        self._update_controls_state()

    def _update_dashboard(self):
        if self.capture is not None and self.capture.isOpened():
            if self.play_state == "paused":
                self.frame = self.last_frame.copy() if self.last_frame is not None else self._create_placeholder_frame()
            else:
                ret, frame = self.capture.read()
                if ret:
                    self.frame = frame
                    self.last_frame = frame.copy()
                    self.play_state = "playing"
                else:
                    self.play_state = "ended"
                    if self.last_frame is not None:
                        self.frame = self.last_frame.copy()
                    else:
                        self.frame = self._create_placeholder_frame()
        else:
            self.frame = self.last_frame.copy() if self.last_frame is not None else self._create_placeholder_frame()
            if self.play_state == "idle":
                self.frame = self._create_placeholder_frame()

        self._update_controls_state()

        color_info = self.tracker.update(self.frame, use_contours=self.use_contours_var.get())
        self._update_mask_probe(color_info)
        face_status, face_coords, faces = self.face_detector.detect(self.frame)

        cx, cy = color_info.get("center_x"), color_info.get("center_y")
        if cx is not None and cy is not None:
            cv2.circle(self.frame, (int(cx), int(cy)), 16, (0, 0, 255), -1)
            cv2.drawMarker(self.frame, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 36, 3)

        if len(faces) > 0:
            fx, fy, fw, fh = map(int, faces[0])
            cv2.rectangle(self.frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        elif face_coords and face_coords != "N/A":
            try:
                parts = str(face_coords).replace("x:", " ").replace("y:", " ").replace("w:", " ").replace("h:", " ").replace(",", " ").split()
                if len(parts) == 4:
                    fx, fy, fw, fh = map(int, parts)
                    cv2.rectangle(self.frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            except Exception:
                pass

        formatted = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(formatted)
        image = self._fit_frame_to_video_area(image)
        photo = ImageTk.PhotoImage(image)
        self.video_area.config(image=photo, compound="center")
        self.video_area.image = photo

        self.target_var.set(f"Target Ref Color: {color_info['bgr_str']}")
        if self.tracker.reference_color is not None:
            blue, green, red = map(int, self.tracker.reference_color)
            self.target_color_preview.config(bg=f"#{red:02x}{green:02x}{blue:02x}")
        else:
            self.target_color_preview.config(bg="#2b3140")
        self.n_var.set(f"N (pixel count): {color_info['pixel_count']}")
        self.sum_x_var.set(f"Sum X (sum x_i): {color_info['sum_x']}")
        self.sum_y_var.set(f"Sum Y (sum y_i): {color_info['sum_y']}")
        center = f"({color_info['center_x']}, {color_info['center_y']})" if color_info['center_x'] is not None else "None"
        self.result_var.set(f"Result (Xc, Yc): {center}")
        self.face_status_var.set(f"Face Status: {face_status}")
        self.face_coords_var.set(f"Coords (x,y,w,h): {face_coords}")

        mask = color_info["binary_mask"]
        if mask.size:
            mask_image = self._fit_mask_to_preview(mask)
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.mask_preview.config(image=mask_photo)
            self.mask_preview.image = mask_photo
        else:
            self.mask_preview.config(image='')

        self.root.after(30, self._update_dashboard)

    def run(self):
        self.root.mainloop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()