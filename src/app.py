import time

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from src.color_tracking import ColorTracker
from src.config import DEFAULT_TOLERANCE, WINDOW_NAME, haar_cascade_path
from src.haar_cascade_inspector import HaarCascadeInspector
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
        self.haar_inspector = HaarCascadeInspector(haar_cascade_path())
        self.current_feature_index = self.haar_inspector.get_default_feature_index()
        self.current_candidate_index = 0
        self.capture = None
        self.frame = self._create_placeholder_frame()
        self.last_frame = None
        self.video_path = None
        self.source_type = None
        self.play_state = "idle"
        self.mask_probe_point = None
        self.last_face_analysis = None
        self.last_color_info = None
        self.active_sidebar_panel = "centroid"
        self.last_frame_time = None
        self.current_fps = 0.0

        self._build_ui()
        self._set_sidebar_panel("centroid")
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

    def _get_face_preview_geometry(self, src_width, src_height):
        target_width = self.face_preview.winfo_width() or 380
        target_height = self.face_preview.winfo_height() or 220

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

    def _fit_face_to_preview(self, image):
        src_width, src_height = image.size
        target_width, target_height, resized_width, resized_height, offset_x, offset_y = self._get_face_preview_geometry(src_width, src_height)

        resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        canvas.paste(resized_image, (offset_x, offset_y))
        return canvas

    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=5)
        self.root.grid_columnconfigure(1, weight=4, minsize=560)

        left_panel = tk.Frame(self.root, bg="#0d1117")
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.grid_rowconfigure(0, weight=0)
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_rowconfigure(2, weight=0)
        left_panel.grid_columnconfigure(0, weight=1)

        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.fps_label = tk.Label(
            left_panel,
            textvariable=self.fps_var,
            bg="#0d1117",
            fg="#dfe8ff",
            font=("Consolas", 12, "bold"),
            anchor="w",
        )
        self.fps_label.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 4))

        self.video_area = tk.Label(
            left_panel,
            bg="#000000",
            highlightthickness=0,
            anchor="center",
            justify="center",
        )
        self.video_area.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
        self.video_area.bind("<Button-1>", self._on_video_click)

        controls_frame = tk.Frame(left_panel, bg="#0d1117")
        controls_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
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

        self.sidebar_tabs = tk.Frame(right_panel, bg="#1a1f2b")
        self.sidebar_tabs.pack(fill="x", padx=20, pady=(15, 6))
        self.sidebar_tabs.grid_columnconfigure(0, weight=1)
        self.sidebar_tabs.grid_columnconfigure(1, weight=1)

        self.centroid_tab = tk.Button(
            self.sidebar_tabs,
            text="Center of Mass",
            command=lambda: self._set_sidebar_panel("centroid"),
            bg="#ff7a00",
            fg="white",
            activebackground="#ff8f1a",
            activeforeground="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
        )
        self.haar_tab = tk.Button(
            self.sidebar_tabs,
            text="Haar Cascade",
            command=lambda: self._set_sidebar_panel("haar"),
            bg="#2b3140",
            fg="white",
            activebackground="#3a4154",
            activeforeground="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
        )
        self.centroid_tab.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.haar_tab.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.centroid_panel = tk.Frame(right_panel, bg="#1a1f2b")
        self.haar_panel = tk.Frame(right_panel, bg="#1a1f2b")
        self.centroid_panel.pack(fill="both", expand=True)
        self.haar_panel.pack(fill="both", expand=True)
        self.haar_panel.pack_forget()

        self.header = tk.Label(
            self.centroid_panel,
            text="=== COLOR TRACKING ===",
            bg="#1a1f2b",
            fg="#ffd000",
            font=("Consolas", 13, "bold"),
            anchor="w",
        )
        self.header.pack(fill="x", padx=24, pady=(0, 6))

        self.tolerance_input_var = tk.StringVar(value=str(self.tracker.tolerance))
        self.target_var = tk.StringVar(value="Target Ref Color: None")
        self.mask_var = tk.StringVar(value="Current Binary Mask:")
        self.use_contours_var = tk.BooleanVar(value=False)
        self.center_title_var = tk.StringVar(value="Center of mass (Xc, Yc)")
        self.n_var = tk.StringVar(value="N (pixel count): 0")
        self.sum_x_var = tk.StringVar(value="Sum X (sum x_i): 0")
        self.sum_y_var = tk.StringVar(value="Sum Y (sum y_i): 0")
        self.result_var = tk.StringVar(value="Result (Xc, Yc): None")
        self.face_title_var = tk.StringVar(value="=== HAAR CASCADE ===")
        self.face_status_var = tk.StringVar(value="Face Status: Not Found")
        self.haar_feature_title_var = tk.StringVar(value="Feature")
        self.candidate_index_var = tk.StringVar(value="Candidate: N/A")
        self.haar_feature_index_var = tk.StringVar(value="Feature: N/A")
        self.haar_feature_info_var = tk.StringVar(value="Feature: N/A")
        self.haar_feature_calc_var = tk.StringVar(value="Feature value: N/A")
        self.haar_feature_xml_threshold_var = tk.StringVar(value="XML threshold: N/A")
        self.haar_feature_threshold_calc_var = tk.StringVar(value="Normalized threshold calc: N/A")
        self.haar_feature_compare_var = tk.StringVar(value="Compare: N/A")
        self.haar_feature_leafs_var = tk.StringVar(value="Leaf values: N/A")
        self.haar_feature_pass_var = tk.StringVar(value="Feature passed:: N/A")
        self.haar_feature_decision_var = tk.StringVar(value="Selected branch: N/A")
        self.haar_stage_title_var = tk.StringVar(value="Stage: N/A")
        self.haar_stage_info_var = tk.StringVar(value="Stage: N/A")
        self.haar_stage_compare_var = tk.StringVar(value="Stage compare: N/A")
        self.haar_stage_pass_var = tk.StringVar(value="Stage passed: N/A")
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

        tolerance_row = tk.Frame(self.centroid_panel, bg="#1a1f2b")
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

        target_row = tk.Frame(self.centroid_panel, bg="#1a1f2b")
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
            self.centroid_panel,
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
        tk.Label(self.centroid_panel, textvariable=self.mask_var, **text_style).pack(anchor="w", padx=24, pady=(10, 6))

        mask_panel = tk.Frame(self.centroid_panel, bg="#1a1f2b")
        mask_panel.pack(anchor="w", padx=24, pady=(0, 10))

        self.mask_preview = tk.Label(mask_panel, bg="#000000", width=380, height=220, relief="solid", borderwidth=2)
        self.mask_preview.pack_propagate(False)
        self.mask_preview.pack(anchor="w", padx=0, pady=(0, 10))
        self.mask_preview.bind("<Button-1>", self._on_mask_preview_click)

        tk.Label(mask_panel, textvariable=self.mask_probe_title_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12, "bold"), anchor="w").pack(anchor="w", padx=0, pady=(0, 2))
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

        tk.Label(self.centroid_panel, textvariable=self.center_title_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(10, 4))
        tk.Label(self.centroid_panel, textvariable=self.n_var, **text_style).pack(anchor="w", padx=24)
        tk.Label(self.centroid_panel, textvariable=self.sum_x_var, **text_style).pack(anchor="w", padx=24)
        tk.Label(self.centroid_panel, textvariable=self.sum_y_var, **text_style).pack(anchor="w", padx=24)
        tk.Label(self.centroid_panel, textvariable=self.result_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(4, 12))

        tk.Label(self.haar_panel, textvariable=self.face_title_var, bg="#1a1f2b", fg="#00ff00", font=("Consolas", 13, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(4, 4))
        self.face_preview = tk.Label(self.haar_panel, bg="#000000", width=380, height=220, relief="solid", borderwidth=2)
        self.face_preview.pack_propagate(False)
        self.face_preview.pack(anchor="w", padx=24, pady=(0, 10))
        candidate_nav_row = tk.Frame(self.haar_panel, bg="#1a1f2b")
        candidate_nav_row.pack(anchor="w", padx=24, pady=(0, 8))
        tk.Button(candidate_nav_row, text="<", command=lambda: self._shift_candidate(-1), bg="#2b3140", fg="white", activebackground="#3a4154", activeforeground="white", font=("Consolas", 12, "bold"), bd=0, relief="flat", width=4, cursor="hand2").pack(side="left")
        tk.Label(candidate_nav_row, textvariable=self.candidate_index_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="center", width=32).pack(side="left", padx=8)
        tk.Button(candidate_nav_row, text=">", command=lambda: self._shift_candidate(1), bg="#2b3140", fg="white", activebackground="#3a4154", activeforeground="white", font=("Consolas", 12, "bold"), bd=0, relief="flat", width=4, cursor="hand2").pack(side="left")
        feature_nav_row = tk.Frame(self.haar_panel, bg="#1a1f2b")
        feature_nav_row.pack(anchor="w", padx=24, pady=(0, 8))
        tk.Button(feature_nav_row, text="<", command=lambda: self._shift_feature(-1), bg="#2b3140", fg="white", activebackground="#3a4154", activeforeground="white", font=("Consolas", 12, "bold"), bd=0, relief="flat", width=4, cursor="hand2").pack(side="left")
        tk.Label(feature_nav_row, textvariable=self.haar_feature_index_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="center", width=32).pack(side="left", padx=8)
        tk.Button(feature_nav_row, text=">", command=lambda: self._shift_feature(1), bg="#2b3140", fg="white", activebackground="#3a4154", activeforeground="white", font=("Consolas", 12, "bold"), bd=0, relief="flat", width=4, cursor="hand2").pack(side="left")
        tk.Label(self.haar_panel, textvariable=self.face_status_var, bg="#1a1f2b", fg="#00ff00", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_feature_title_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(8, 2))

        tk.Label(self.haar_panel, textvariable=self.haar_feature_calc_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_feature_xml_threshold_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_feature_threshold_calc_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_feature_compare_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_feature_leafs_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_feature_pass_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_feature_decision_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24, pady=(0, 12))
        tk.Label(self.haar_panel, textvariable=self.haar_stage_title_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12, "bold"), anchor="w").pack(anchor="w", padx=24, pady=(0, 2))
        tk.Label(self.haar_panel, textvariable=self.haar_stage_info_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_stage_compare_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w", justify="left", wraplength=330).pack(anchor="w", padx=24)
        tk.Label(self.haar_panel, textvariable=self.haar_stage_pass_var, bg="#1a1f2b", fg="#ffffff", font=("Consolas", 12), anchor="w").pack(anchor="w", padx=24, pady=(0, 12))

        self.root.bind("<Escape>", lambda event: self.root.destroy())

    def _set_sidebar_panel(self, panel_name):
        if panel_name not in {"centroid", "haar"}:
            return

        self.active_sidebar_panel = panel_name
        if panel_name == "centroid":
            self.centroid_panel.pack(fill="both", expand=True)
            self.haar_panel.pack_forget()
            self.centroid_tab.config(bg="#ff7a00", activebackground="#ff8f1a")
            self.haar_tab.config(bg="#2b3140", activebackground="#3a4154")
        else:
            self.haar_panel.pack(fill="both", expand=True)
            self.centroid_panel.pack_forget()
            self.haar_tab.config(bg="#ff7a00", activebackground="#ff8f1a")
            self.centroid_tab.config(bg="#2b3140", activebackground="#3a4154")

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

    def _shift_feature(self, step):
        feature_count = self.haar_inspector.get_feature_count()
        if feature_count == 0:
            self.current_feature_index = 0
            return

        self.current_feature_index = (self.current_feature_index + step) % feature_count

    def _shift_candidate(self, step):
        self.current_candidate_index += step

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
                f"Threshold check: {'Feature passed:' if raw_pass else 'Feature failed'}"
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

        self.last_face_analysis = None

        self._update_controls_state()

    def _update_dashboard(self):
        now = time.perf_counter()
        if self.capture is not None and self.capture.isOpened():
            if self.play_state == "paused":
                self.frame = self.last_frame.copy() if self.last_frame is not None else self._create_placeholder_frame()
                self.current_fps = 0.0
            else:
                ret, frame = self.capture.read()
                if ret:
                    if self.last_frame_time is not None:
                        elapsed = now - self.last_frame_time
                        self.current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    else:
                        self.current_fps = 0.0
                    self.last_frame_time = now
                    self.frame = frame
                    self.last_frame = frame.copy()
                    self.play_state = "playing"
                else:
                    self.play_state = "ended"
                    self.current_fps = 0.0
                    if self.last_frame is not None:
                        self.frame = self.last_frame.copy()
                    else:
                        self.frame = self._create_placeholder_frame()
                    self.last_frame_time = None
        else:
            self.frame = self.last_frame.copy() if self.last_frame is not None else self._create_placeholder_frame()
            if self.play_state == "idle":
                self.frame = self._create_placeholder_frame()
            self.current_fps = 0.0
            self.last_frame_time = None

        self.fps_var.set(f"FPS: {self.current_fps:.1f}")
        self._update_controls_state()

        centroid_active = self.active_sidebar_panel == "centroid"
        haar_active = self.active_sidebar_panel == "haar"

        if centroid_active:
            color_info = self.tracker.update(self.frame, use_contours=self.use_contours_var.get())
            self.last_color_info = color_info
            self._update_mask_probe(color_info)
            cx, cy = color_info.get("center_x"), color_info.get("center_y")
            if cx is not None and cy is not None:
                cv2.circle(self.frame, (int(cx), int(cy)), 16, (0, 0, 255), -1)
                cv2.drawMarker(self.frame, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 36, 3)

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

            mask = color_info["binary_mask"]
            if mask.size:
                mask_image = self._fit_mask_to_preview(mask)
                mask_photo = ImageTk.PhotoImage(mask_image)
                self.mask_preview.config(image=mask_photo)
                self.mask_preview.image = mask_photo
            else:
                self.mask_preview.config(image='')
        else:
            self._update_mask_probe({"binary_mask": np.zeros((1, 1), dtype=np.uint8)})
            self.target_var.set("Target Ref Color: disabled")
            self.n_var.set("N (pixel count): disabled")
            self.sum_x_var.set("Sum X (sum x_i): disabled")
            self.sum_y_var.set("Sum Y (sum y_i): disabled")
            self.result_var.set("Result (Xc, Yc): disabled")
            self.mask_preview.config(image='')
            self.target_color_preview.config(bg="#2b3140")

        if haar_active:
            if self.play_state == "paused" and self.last_face_analysis is not None:
                face_status, face_coords, faces, candidates = self.last_face_analysis
            else:
                face_status, face_coords, faces, candidates = self.face_detector.detect(self.frame)
                candidates = sorted(candidates, key=lambda rect: (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))
                self.last_face_analysis = (face_status, face_coords, faces, candidates)
            face_region = None
            selected_candidate = None
            candidate_count = len(candidates)
            if candidate_count > 0:
                self.current_candidate_index = self.current_candidate_index % candidate_count
                selected_candidate = tuple(map(int, candidates[self.current_candidate_index]))
                face_region = selected_candidate
                self.candidate_index_var.set(f"Candidate: {self.current_candidate_index + 1} / {candidate_count}")
            else:
                self.current_candidate_index = 0
                self.candidate_index_var.set("Candidate: N/A")
                if len(faces) > 0:
                    fx, fy, fw, fh = map(int, faces[0])
                    face_region = (fx, fy, fw, fh)

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

            self.face_status_var.set(f"Face Status: {face_status}")
            face_image = self.haar_inspector.render_feature_overlay(self.frame.copy(), self.current_feature_index, face_region)
            face_preview_bgr = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
            frame_h, frame_w = face_preview_bgr.shape[:2]
            _, _, preview_w, preview_h, _, _ = self._get_face_preview_geometry(frame_w, frame_h)
            preview_scale = min(preview_w / max(frame_w, 1), preview_h / max(frame_h, 1))
            preview_rect_thickness = max(2, int(round(2 / max(preview_scale, 1e-6))))
            if len(faces) > 0:
                fx, fy, fw, fh = map(int, faces[0])
                cv2.rectangle(face_preview_bgr, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), preview_rect_thickness)
            if selected_candidate is not None:
                cx, cy, cw, ch = selected_candidate
                cv2.rectangle(face_preview_bgr, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), preview_rect_thickness)
            face_image = Image.fromarray(cv2.cvtColor(face_preview_bgr, cv2.COLOR_BGR2RGB))
            face_image = self._fit_face_to_preview(face_image)
            face_photo = ImageTk.PhotoImage(face_image)
            self.face_preview.config(image=face_photo)
            self.face_preview.image = face_photo

            feature_count = self.haar_inspector.get_feature_count()
            if feature_count > 0:
                effective_index = self.current_feature_index % feature_count
                self.current_feature_index = effective_index
                haar_result = self.haar_inspector.evaluate_feature(self.frame.copy(), effective_index, face_region)
                self.haar_feature_title_var.set(f"Feature #{effective_index + 1}")
                self.haar_feature_index_var.set(f"Feature: {effective_index + 1} / {feature_count}")
                self.haar_feature_info_var.set("")
            else:
                haar_result = {
                    "weighted_sum": 0.0,
                    "normalization": 1.0,
                    "variance_norm_factor": 1.0,
                    "effective_threshold": 0.0,
                    "response": 0.0,
                    "threshold": 0.0,
                    "passed": False,
                    "left_value": 0.0,
                    "right_value": 0.0,
                    "selected_branch": "N/A",
                    "selected_leaf_value": 0.0,
                    "stage_index": None,
                    "stage_sum": 0.0,
                    "stage_threshold": 0.0,
                    "stage_passed": False,
                    "matched": False,
                }
                self.haar_feature_title_var.set("Feature: N/A")
                self.haar_feature_index_var.set("Feature: N/A")
                self.haar_feature_info_var.set("")

            self.haar_feature_calc_var.set(
                f"Weighted sum (raw): {haar_result['weighted_sum']:.2f}\nNormalization area: {haar_result['normalization']:.2f}\nFeature value: {haar_result['weighted_sum']:.2f} / {haar_result['normalization']:.2f} = {haar_result['response']:.6f}\nVariance normalization factor: {haar_result['variance_norm_factor']:.6f}"
            )
            self.haar_feature_xml_threshold_var.set(
                f"XML threshold: {haar_result['threshold']:.6f}"
            )
            self.haar_feature_threshold_calc_var.set(
                f"Normalized threshold calc: {haar_result['threshold']:.6f} * {haar_result['variance_norm_factor']:.6f} = {haar_result['effective_threshold']:.6f}"
            )
            compare_sign = ">=" if haar_result["passed"] else "<"
            self.haar_feature_compare_var.set(
                f"Compare: {haar_result['response']:.6f} {compare_sign} {haar_result['effective_threshold']:.6f}"
            )
            self.haar_feature_leafs_var.set(
                f"Leaf values: left={haar_result['left_value']:.6f} | right={haar_result['right_value']:.6f}"
            )
            self.haar_feature_pass_var.set(f"Feature passed: {haar_result['passed']}")
            if haar_result.get("matched"):
                decision = haar_result["selected_branch"]
                leaf_value = haar_result["selected_leaf_value"]
                self.haar_feature_decision_var.set(
                    f"Selected branch: {decision} | selected leaf value: {leaf_value:.6f}"
                )
            else:
                self.haar_feature_decision_var.set("Selected branch: N/A")

            if haar_result.get("stage_index") is not None:
                stage_display = int(haar_result["stage_index"]) + 1
                self.haar_stage_title_var.set(f"Stage #{stage_display}")
                self.haar_stage_info_var.set(
                    f"sum={haar_result['stage_sum']:.6f}, threshold={haar_result['stage_threshold']:.6f}"
                )
                stage_sign = ">=" if haar_result["stage_passed"] else "<"
                self.haar_stage_compare_var.set(
                    f"Stage compare: {haar_result['stage_sum']:.6f} {stage_sign} {haar_result['stage_threshold']:.6f}"
                )
                self.haar_stage_pass_var.set(f"Stage passed: {haar_result['stage_passed']}")
            else:
                self.haar_stage_title_var.set("Stage: N/A")
                self.haar_stage_info_var.set("sum=N/A, threshold=N/A")
                self.haar_stage_compare_var.set("Stage compare: N/A")
                self.haar_stage_pass_var.set("Stage passed: N/A")
        else:
            self.face_status_var.set("Face Status: disabled")
            self.candidate_index_var.set("Candidate: disabled")
            self.haar_feature_title_var.set("Feature: disabled")
            self.haar_feature_index_var.set("Feature: disabled")
            self.haar_feature_calc_var.set("Feature value: disabled")
            self.haar_feature_xml_threshold_var.set("XML threshold: disabled")
            self.haar_feature_threshold_calc_var.set("Normalized threshold calc: disabled")
            self.haar_feature_compare_var.set("Compare: disabled")
            self.haar_feature_leafs_var.set("Leaf values: disabled")
            self.haar_feature_pass_var.set("Feature passed: disabled")
            self.haar_feature_decision_var.set("Selected branch: disabled")
            self.haar_stage_title_var.set("Stage: disabled")
            self.haar_stage_info_var.set("Stage: disabled")
            self.haar_stage_compare_var.set("Stage compare: disabled")
            self.haar_stage_pass_var.set("Stage passed: disabled")
            self.face_preview.config(image='')

        formatted = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(formatted)
        image = self._fit_frame_to_video_area(image)
        photo = ImageTk.PhotoImage(image)
        self.video_area.config(image=photo, compound="center")
        self.video_area.image = photo

        self.root.after(30, self._update_dashboard)

    def run(self):
        self.root.mainloop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()