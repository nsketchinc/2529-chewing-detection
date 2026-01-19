"""Wrapper for MediaPipe FaceLandmarker (Tasks API).

Responsibility: manage model lifecycle and run detection.
Metrics計算は呼び出し側に任せる。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class DetectionResult:
    """Detection outputs for a single frame."""

    normalized_landmarks: Optional[np.ndarray]
    task_result: vision.FaceLandmarkerResult


class FaceLandmarkerRunner:
    """Manage FaceLandmarker lifecycle and per-frame detection."""

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        frame_interval_ms: int = 33,
    ) -> None:
        self.frame_interval_ms = frame_interval_ms
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker: Optional[vision.FaceLandmarker] = None
        self._timestamp_ms: int = 0

    def __enter__(self) -> "FaceLandmarkerRunner":
        self._landmarker = vision.FaceLandmarker.create_from_options(self.options)
        self._timestamp_ms = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(self, frame_bgr) -> DetectionResult:
        """Run face landmark detection on a BGR frame and return normalized landmarks."""
        if self._landmarker is None:
            raise RuntimeError("FaceLandmarkerRunner must be used as a context manager")

        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += self.frame_interval_ms
        if result.face_landmarks:
            frame_height, frame_width = frame_bgr.shape[:2]
            landmarks = self._landmarks_to_array(result.face_landmarks[0])
            return DetectionResult(normalized_landmarks=landmarks, task_result=result)
        return DetectionResult(normalized_landmarks=None, task_result=result)

    def _landmarks_to_array(self, face_landmarks) -> np.ndarray:
        data = [[], [], []]
        for landmark in face_landmarks:
            data[0].append(landmark.x)
            data[1].append(landmark.y)
            data[2].append(landmark.z)
        return np.array(data)
