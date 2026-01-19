"""Landmark-based distance calculations for face metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence

import numpy as np


@dataclass
class LandmarkMeasure:
    cheek_left: Tuple[int, int]
    cheek_right: Tuple[int, int]
    jaw_tip: Tuple[int, int]
    cheek_width: float
    cheek_jaw_gap: float
    gap_diff: float


class LandmarkMetrics:
    """Compute face distances from normalized landmarks."""

    def __init__(self, jaw_indices: Sequence[int] | None = None) -> None:
        self.jaw_indices = list(jaw_indices) if jaw_indices is not None else [187, 214, 210, 211, 32, 208, 199, 428, 262, 431, 430, 434, 411]

    def measure(self, landmarks: np.ndarray, frame_width: int, frame_height: int) -> LandmarkMeasure:
        """Calculate key distances using normalized landmarks.

        Args:
            landmarks: np.array shape (3, N) with normalized x,y,z.
            frame_width: image width in pixels.
            frame_height: image height in pixels.
        """
        points = self._get_key_points(landmarks, frame_width, frame_height)
        cheek_width, _, _ = self.measure_cheek_width(landmarks, frame_width, frame_height)
        cheek_jaw_gap, _, _ = self.measure_cheek_jaw_gap(landmarks, frame_width, frame_height)
        gap_diff = self.measure_gap_diff(cheek_width, cheek_jaw_gap)

        return LandmarkMeasure(
            cheek_left=points["cheek_left"],
            cheek_right=points["cheek_right"],
            jaw_tip=points["jaw_tip"],
            cheek_width=cheek_width,
            cheek_jaw_gap=cheek_jaw_gap,
            gap_diff=gap_diff,
        )

    def measure_cheek_width(self, landmarks: np.ndarray, frame_width: int, frame_height: int) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        """Measure horizontal distance between left and right cheek points."""
        points = self._get_key_points(landmarks, frame_width, frame_height)
        pt_left = points["cheek_left"]
        pt_right = points["cheek_right"]
        cheek_width = float(np.linalg.norm(np.array(pt_left) - np.array(pt_right)))
        return cheek_width, pt_left, pt_right

    def measure_cheek_jaw_gap(self, landmarks: np.ndarray, frame_width: int, frame_height: int) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        """Measure vertical gap between left cheek and jaw tip."""
        points = self._get_key_points(landmarks, frame_width, frame_height)
        pt_left = points["cheek_left"]
        jaw_tip = points["jaw_tip"]
        cheek_jaw_gap = float(abs(pt_left[1] - jaw_tip[1]))
        return cheek_jaw_gap, pt_left, jaw_tip

    def measure_gap_diff(self, cheek_width: float, cheek_jaw_gap: float) -> float:
        """Measure difference between cheek width and cheek-jaw gap."""
        return float(abs(cheek_width - cheek_jaw_gap))

    def _to_pixel(self, landmarks: np.ndarray, idx: int, width: int, height: int) -> Tuple[int, int]:
        x = int(landmarks[0][idx] * width)
        y = int(landmarks[1][idx] * height)
        return x, y

    def _get_key_points(self, landmarks: np.ndarray, width: int, height: int) -> dict:
        cheek_left_idx = self.jaw_indices[0]
        jaw_tip_idx = self.jaw_indices[6]
        cheek_right_idx = self.jaw_indices[12]

        return {
            "cheek_left": self._to_pixel(landmarks, cheek_left_idx, width, height),
            "jaw_tip": self._to_pixel(landmarks, jaw_tip_idx, width, height),
            "cheek_right": self._to_pixel(landmarks, cheek_right_idx, width, height),
        }
