"""Landmark-based distance calculations for face metrics (pixel coordinates expected)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple 

import numpy as np


# あごのデフォルトの特徴点インデックス（MediaPipe Face Meshのものを使用）
JAW_INDICES = [187, 214, 210, 211, 32, 208, 199, 428, 262, 431, 430, 434, 411]

CHEEK_LEFT_INDEX = 187
CHEEK_RIGHT_INDEX = 411
JAW_TIP_INDEX = 199


class LandmarkMetrics:
    """Compute face distances from pixel-based landmarks."""

    def __init__(self) -> None:
      return

    def get_jaw_indices(self) -> list[int]:
        return JAW_INDICES
      
    def get_cheek_left_index(self) -> int:
        return CHEEK_LEFT_INDEX
    
    def get_cheek_right_index(self) -> int:
        return CHEEK_RIGHT_INDEX
    
    def get_jaw_tip_index(self) -> int:
        return JAW_TIP_INDEX

    def measure_cheek_width(self, landmarks_px: np.ndarray) -> float:
        """Measure horizontal distance between left and right cheek points using pixel coords."""
        pt_left = self._point(landmarks_px, CHEEK_LEFT_INDEX)
        pt_right = self._point(landmarks_px, CHEEK_RIGHT_INDEX)
        cheek_width = float(np.linalg.norm(np.array(pt_left) - np.array(pt_right)))
        return cheek_width

    def measure_cheek_jaw_gap(self, landmarks_px: np.ndarray) -> float:
        """Measure vertical gap between left cheek and jaw tip using pixel coords."""
        pt_left = self._point(landmarks_px, CHEEK_LEFT_INDEX)
        jaw_tip = self._point(landmarks_px, JAW_TIP_INDEX)
        cheek_jaw_gap = float(abs(pt_left[1] - jaw_tip[1]))
        return cheek_jaw_gap

    def _point(self, landmarks_px: np.ndarray, idx: int) -> Tuple[int, int]:
        x = int(landmarks_px[0][idx])
        y = int(landmarks_px[1][idx])
        return x, y
    