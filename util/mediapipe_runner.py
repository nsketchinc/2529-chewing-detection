"""Wrapper for MediaPipe FaceLandmarker (Tasks API).

Responsibility: manage model lifecycle and run detection.
"""
from __future__ import annotations

from ast import Tuple
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class MediapipeRunner:
    """Manage FaceLandmarker lifecycle and per-frame detection."""

    def __init__(
        self,
        model_path: str = "data/face_landmarker.task",
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

    def __enter__(self) -> "MediapipeRunner":
        self._landmarker = vision.FaceLandmarker.create_from_options(self.options)
        self._timestamp_ms = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(self, rgb_image) -> DetectionResult:
        """Run face landmark detection on an RGB frame and return normalized landmarks only."""
        if self._landmarker is None:
            raise RuntimeError("MediapipeRunner must be used as a context manager")

        # process image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        frame_height, frame_width = rgb_image.shape[:2]

        # result
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += self.frame_interval_ms

        # if no result
        if result.face_landmarks is None:
            return DetectionResult(landmarks=None, frame_height=frame_height, frame_width=frame_width)

        landmarks = self._landmarks_to_array(result.face_landmarks[0])
        return DetectionResult(landmarks=landmarks, frame_height=frame_height, frame_width=frame_width)


    def _landmarks_to_array(self, face_landmarks) -> np.ndarray:
        data = [[], [], []]
        for landmark in face_landmarks:
            data[0].append(landmark.x)
            data[1].append(landmark.y)
            data[2].append(landmark.z)
        return np.array(data)


class DetectionResult:
    """Detection outputs for a single frame."""

    landmarks: Optional[np.ndarray]

    def __init__(self, landmarks: Optional[np.ndarray], frame_height: int, frame_width: int) -> None:
        self.landmarks = landmarks
        self.frame_height = frame_height
        self.frame_width = frame_width

    def has_landmarks(self) -> bool:
        return self.landmarks is not None
    
    def get_normalized_landmarks(self) -> np.ndarray:
        if self.landmarks is None:
            raise ValueError("No landmarks available")
        return self.landmarks

    def get_pixel_landmarks(self) -> np.ndarray:
        if self.landmarks is None:
            raise ValueError("No landmarks available")
        pixel_landmarks = np.vstack([
            self.landmarks[0] * self.frame_width,
            self.landmarks[1] * self.frame_height,
        ])
        return pixel_landmarks

    def get_normalized_point(self, index: int) -> Tuple[float, float]:
        if self.landmarks is None:
            raise ValueError("No landmarks available")
        x = self.landmarks[0][index]
        y = self.landmarks[1][index]
        return (x, y)

    def get_pixel_point(self, index: int) -> Tuple[int, int]:
        if self.landmarks is None:
            raise ValueError("No landmarks available")
        x = int(self.landmarks[0][index] * self.frame_width)
        y = int(self.landmarks[1][index] * self.frame_height)
        return (x, y)
