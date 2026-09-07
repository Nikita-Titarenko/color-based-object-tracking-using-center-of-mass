from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class HaarRect:
    x: float
    y: float
    width: float
    height: float
    weight: float


@dataclass(frozen=True)
class HaarFeature:
    rects: tuple[HaarRect, ...]


@dataclass(frozen=True)
class HaarWeakClassifier:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float


@dataclass(frozen=True)
class HaarStage:
    threshold: float
    weak_classifiers: tuple[HaarWeakClassifier, ...]


class HaarCascadeInspector:
    def __init__(self, cascade_path):
        self.cascade_path = Path(cascade_path)
        self.base_width = 24
        self.base_height = 24
        self.features = self._load_features()
        self.stages = self._load_stages()

    def _parse_numbers(self, text):
        return [float(value) for value in text.split()]

    def _load_cascade_root(self):
        tree = ET.parse(self.cascade_path)
        root = tree.getroot().find("cascade")
        if root is None:
            raise ValueError("Invalid Haar cascade XML: missing cascade node")

        width_text = root.findtext("width")
        height_text = root.findtext("height")
        if width_text is not None:
            self.base_width = int(width_text)
        if height_text is not None:
            self.base_height = int(height_text)

        return root

    def _load_features(self):
        root = self._load_cascade_root()
        features_node = root.find("features")
        if features_node is None:
            return tuple()

        features = []
        for feature_node in features_node:
            rects_node = feature_node.find("rects")
            if rects_node is None:
                continue

            rects = []
            for rect_node in rects_node:
                values = self._parse_numbers(rect_node.text or "")
                if len(values) != 5:
                    continue
                x, y, width, height, weight = values
                rects.append(HaarRect(x, y, width, height, weight))

            if rects:
                features.append(HaarFeature(tuple(rects)))

        return tuple(features)

    def _load_stages(self):
        root = self._load_cascade_root()
        stages_node = root.find("stages")
        if stages_node is None:
            return tuple()

        stages = []
        for stage_node in stages_node:
            threshold = float(stage_node.findtext("stageThreshold") or 0.0)
            weak_classifiers_node = stage_node.find("weakClassifiers")
            weak_classifiers = []

            if weak_classifiers_node is not None:
                for weak_node in weak_classifiers_node:
                    internal_values = self._parse_numbers(weak_node.findtext("internalNodes") or "")
                    leaf_values = self._parse_numbers(weak_node.findtext("leafValues") or "")
                    if len(internal_values) < 4 or len(leaf_values) < 2:
                        continue

                    feature_index = int(internal_values[2])
                    weak_threshold = float(internal_values[3])
                    weak_classifiers.append(
                        HaarWeakClassifier(
                            feature_index=feature_index,
                            threshold=weak_threshold,
                            left_value=float(leaf_values[0]),
                            right_value=float(leaf_values[1]),
                        )
                    )

            stages.append(HaarStage(threshold=threshold, weak_classifiers=tuple(weak_classifiers)))

        return tuple(stages)

    def get_feature_count(self):
        return len(self.features)

    def get_feature(self, feature_index):
        if not self.features:
            return None

        feature_index = max(0, min(feature_index, len(self.features) - 1))
        return self.features[feature_index]

    def get_default_feature_index(self):
        if not self.stages:
            return 0
        if not self.stages[0].weak_classifiers:
            return 0
        return max(0, min(self.stages[0].weak_classifiers[0].feature_index, len(self.features) - 1))

    def get_weak_classifier_for_feature(self, feature_index):
        if not self.stages or not self.features:
            return None, None

        feature_index = max(0, min(feature_index, len(self.features) - 1))
        for stage_index, stage in enumerate(self.stages):
            for weak in stage.weak_classifiers:
                if weak.feature_index == feature_index:
                    return weak, stage_index

        return None, None

    def _sum_rect(self, integral, left, top, right, bottom):
        return (
            integral[bottom, right]
            - integral[top, right]
            - integral[bottom, left]
            + integral[top, left]
        )

    def _compute_variance_norm_factor(self, integral, squared_integral, width, height):
        area = max(width * height, 1)
        inv_area = 1.0 / float(area)
        total = self._sum_rect(integral, 0, 0, width, height)
        total_sq = self._sum_rect(squared_integral, 0, 0, width, height)
        mean = total * inv_area
        variance = max(total_sq * inv_area - mean * mean, 0.0)
        return max(float(np.sqrt(variance)), 1.0)

    def _compute_feature_response_from_integral(self, feature_index, integral, width, height):
        feature = self.get_feature(feature_index)
        if feature is None:
            return 0.0

        scale_x = width / self.base_width
        scale_y = height / self.base_height
        weighted_sum = 0.0
        area = max(width * height, 1)
        inv_area = 1.0 / float(area)

        for rect in feature.rects:
            left = int(round(rect.x * scale_x))
            top = int(round(rect.y * scale_y))
            right = int(round((rect.x + rect.width) * scale_x))
            bottom = int(round((rect.y + rect.height) * scale_y))

            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(left + 1, min(right, width))
            bottom = max(top + 1, min(bottom, height))

            rect_sum = self._sum_rect(integral, left, top, right, bottom)
            weighted_sum += rect.weight * rect_sum

        return weighted_sum * inv_area

    def _evaluate_weak_classifier(self, weak_classifier, integral, squared_integral, width, height):
        response = self._compute_feature_response_from_integral(
            weak_classifier.feature_index,
            integral,
            width,
            height,
        )
        variance_norm_factor = self._compute_variance_norm_factor(integral, squared_integral, width, height)
        effective_threshold = weak_classifier.threshold * variance_norm_factor
        passed = response >= effective_threshold
        selected_branch = "right" if passed else "left"
        selected_leaf_value = weak_classifier.right_value if passed else weak_classifier.left_value
        return {
            "response": response,
            "passed": passed,
            "selected_branch": selected_branch,
            "selected_leaf_value": selected_leaf_value,
            "effective_threshold": effective_threshold,
            "variance_norm_factor": variance_norm_factor,
        }

    def render_feature_template(self, feature_index):
        feature = self.get_feature(feature_index)
        if feature is None:
            return Image.fromarray(np.zeros((self.base_height, self.base_width, 3), dtype=np.uint8))

        canvas = np.full((self.base_height, self.base_width, 3), 190, dtype=np.uint8)

        for rect in feature.rects:
            x = int(round(rect.x))
            y = int(round(rect.y))
            width = max(1, int(round(rect.width)))
            height = max(1, int(round(rect.height)))
            color = (255, 255, 255) if rect.weight > 0 else (0, 0, 0)
            cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), color, thickness=-1)
            cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (80, 80, 80), thickness=1)

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def render_feature_overlay(self, frame, feature_index, region=None, alpha=0.45):
        feature = self.get_feature(feature_index)
        if feature is None:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        overlay = frame.copy()
        frame_height, frame_width = overlay.shape[:2]

        if region is None:
            region = (0, 0, frame_width, frame_height)

        x, y, width, height = map(int, region)
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        width = max(1, min(width, frame_width - x))
        height = max(1, min(height, frame_height - y))

        scale_x = width / self.base_width
        scale_y = height / self.base_height

        for rect in feature.rects:
            left = x + int(round(rect.x * scale_x))
            top = y + int(round(rect.y * scale_y))
            right = x + int(round((rect.x + rect.width) * scale_x))
            bottom = y + int(round((rect.y + rect.height) * scale_y))

            left = max(x, min(left, x + width - 1))
            top = max(y, min(top, y + height - 1))
            right = max(left + 1, min(right, x + width))
            bottom = max(top + 1, min(bottom, y + height))

            color = (255, 255, 255) if rect.weight > 0 else (0, 0, 0)
            cv2.rectangle(overlay, (left, top), (right, bottom), color, thickness=-1)
            cv2.rectangle(overlay, (left, top), (right, bottom), (90, 90, 90), thickness=1)

        cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (0, 255, 255), thickness=1)
        blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

    def evaluate_feature(self, frame, feature_index, region=None):
        feature = self.get_feature(feature_index)
        if feature is None:
            return {
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
                "feature_index": 0,
                "region": (0, 0, 0, 0),
                "matched": False,
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = gray.shape[:2]

        if region is None:
            region = (0, 0, frame_width, frame_height)

        x, y, width, height = map(int, region)
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        width = max(1, min(width, frame_width - x))
        height = max(1, min(height, frame_height - y))

        roi = gray[y : y + height, x : x + width]
        if roi.size == 0:
            return {
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
                "feature_index": feature_index,
                "region": (x, y, width, height),
                "matched": False,
            }

        roi_float = roi.astype(np.float64)
        integral = cv2.integral(roi_float)
        squared_integral = cv2.integral(roi_float * roi_float)
        response = self._compute_feature_response_from_integral(feature_index, integral, width, height)
        normalization = max(float(width * height), 1.0)
        weighted_sum = response * normalization
        weak, stage_index = self.get_weak_classifier_for_feature(feature_index)
        variance_norm_factor = self._compute_variance_norm_factor(integral, squared_integral, width, height)

        threshold = weak.threshold if weak is not None else 0.0
        effective_threshold = threshold * variance_norm_factor if weak is not None else 0.0
        passed = response >= effective_threshold if weak is not None else False
        selected_branch = "right" if passed else "left"
        selected_leaf_value = weak.right_value if passed else weak.left_value if weak is not None else 0.0

        stage_sum = 0.0
        stage_threshold = 0.0
        stage_passed = False
        if stage_index is not None:
            stage = self.stages[stage_index]
            stage_threshold = stage.threshold
            for candidate_weak in stage.weak_classifiers:
                weak_eval = self._evaluate_weak_classifier(candidate_weak, integral, squared_integral, width, height)
                stage_sum += weak_eval["selected_leaf_value"]
            stage_passed = stage_sum >= stage_threshold

        return {
            "weighted_sum": weighted_sum,
            "normalization": normalization,
            "variance_norm_factor": variance_norm_factor,
            "effective_threshold": effective_threshold,
            "response": response,
            "threshold": threshold,
            "passed": passed,
            "left_value": weak.left_value if weak is not None else 0.0,
            "right_value": weak.right_value if weak is not None else 0.0,
            "selected_branch": selected_branch if weak is not None else "N/A",
            "selected_leaf_value": selected_leaf_value,
            "feature_index": feature_index,
            "stage_index": stage_index,
            "stage_sum": stage_sum,
            "stage_threshold": stage_threshold,
            "stage_passed": stage_passed,
            "matched": weak is not None,
            "region": (x, y, width, height),
        }