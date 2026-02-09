"""Chewing detection logic based on landmark movements.

Responsibility: Detect first bite and subsequent chewing actions.
"""
from __future__ import annotations

import numpy as np


class ChewingDetector:
    """Detects chewing patterns from mouth and jaw landmark movements."""

    def __init__(
        self,
        sequence_length: int = 30,
        firstbite_threshold: float = 0.15,
        mouth_gap_threshold: float = 15.0,
        min_face_size_threshold: float = 0.0,
        non_continuous_sounds: int = 5,
        face_direction_x_threshold: float = 30.0,
        face_direction_y_threshold: tuple[float, float] = (-20.0, 20.0),
    ) -> None:
        """Initialize chewing detector with configurable thresholds.
        
        Args:
            sequence_length: Number of frames to keep in history
            firstbite_threshold: Threshold for first bite detection
            mouth_gap_threshold: Max mouth opening allowed to count as chewing
            min_face_size_threshold: If > 0, disable chewing detection when face size is smaller than this (px)
            non_continuous_sounds: Minimum frames between sound triggers
            face_direction_x_threshold: Max horizontal face angle deviation
            face_direction_y_threshold: Min/max vertical face angle range
        """
        self.sequence_length = sequence_length
        self.firstbite_threshold = firstbite_threshold
        self.mouth_gap_threshold = mouth_gap_threshold
        self.min_face_size_threshold = min_face_size_threshold
        self.non_continuous_sounds = non_continuous_sounds
        self.face_direction_x_threshold = face_direction_x_threshold
        self.face_direction_y_threshold = face_direction_y_threshold

        # State arrays
        self.flag_list = np.array([0] * sequence_length)
        self.face_state_list = np.array([0] * sequence_length)
        
        # Counters
        self.munching_count = 0
        self.face_state = 0
        self.is_chewing = False

        # Latest face size measurements (pixel space)
        self.last_face_size: float | None = None
        self.last_face_bbox: tuple[float, float, float, float] | None = None
        self.last_face_width: float | None = None
        self.last_face_height: float | None = None

    def detect_chewing(
        self,
        landmarks_px: np.ndarray,
        face_direction_x: float,
        face_direction_y: float,
        prediction_scores: np.ndarray | None = None,
    ) -> tuple[int, float, int]:
        """Detect chewing action from current frame.
        
        Args:
            landmarks_px: Pixel coordinates shaped (2, N) or (3, N). Uses x,y rows.
            face_direction_x: Horizontal face angle
            face_direction_y: Vertical face angle
            prediction_scores: Optional ML prediction scores [first_bite, onwards, other]
            
        Returns:
            flag: 0=none, 1=first_bite, 2=chewing
            mouth_gap: Current mouth opening distance
            face_state: 0=ok, 1=no_face, 2=face_turned, 3=other
        """
        # Measure current mouth gap (for visualization)
        mouth_gap = self._measure_mouth_gap(landmarks_px)

        # Measure face size (for normalization / debugging / downstream use)
        face_size = self._measure_face_size(landmarks_px)

        # If face is too small (far away / unreliable), disable chewing judgement.
        if self.min_face_size_threshold > 0.0:
            if (not np.isfinite(face_size)) or (face_size < self.min_face_size_threshold):
                flag = 0
                self.is_chewing = False

                self.flag_list[:-1] = self.flag_list[1:]
                self.flag_list[-1] = flag

                self.face_state_list[:-1] = self.face_state_list[1:]
                self.face_state_list[-1] = self.face_state

                return flag, mouth_gap, self.face_state
        
        # Evaluate face direction
        # self._evaluate_face_direction(face_direction_x, face_direction_y)

        # Determine chewing flag based primarily on ML scores
        flag = 0
        if prediction_scores is not None:
            if self.face_state in [1, 2]:
                flag = 0
                self.is_chewing = False
            elif mouth_gap > self.mouth_gap_threshold:
                flag = 0
                self.is_chewing = False
            else:
                first_bite_score = prediction_scores[0]
                if self.is_chewing:
                    if first_bite_score > self.firstbite_threshold:
                        flag = 2
                    else:
                        flag = 0
                        self.is_chewing = False
                else:
                    if first_bite_score > self.firstbite_threshold:
                        flag = 1
                        self.is_chewing = True
                        self.munching_count += 1
                    else:
                        flag = 0
                        self.is_chewing = False
        
        self.flag_list[:-1] = self.flag_list[1:]
        self.flag_list[-1] = flag

        self.face_state_list[:-1] = self.face_state_list[1:]
        self.face_state_list[-1] = self.face_state

        return flag, mouth_gap, self.face_state

    def _measure_mouth_gap(self, landmarks_px: np.ndarray) -> float:
        """Measure vertical mouth opening distance for visualization."""
        # Landmark indices for mouth measurements
        mouth_above_idx = 12
        mouth_under_idx = 15
        
        y_coords = landmarks_px[1]
        
        # Vertical mouth gap
        distance_upper_under = float(np.abs(y_coords[mouth_above_idx] - y_coords[mouth_under_idx]))
        return distance_upper_under

    def _measure_face_size(self, landmarks_px: np.ndarray) -> float:
        """Measure an approximate face size from pixel landmarks.

        The primary output is a scale value in pixels that can be used to
        normalize other distances (e.g. mouth gap). We compute:

        - face bbox (min/max over all landmarks)
        - face width/height from bbox
        - face size: bbox diagonal length

        Returns:
            face_size_px: Face scale in pixels (bbox diagonal). Returns NaN if input is invalid.
        """
        if (
            not isinstance(landmarks_px, np.ndarray)
            or landmarks_px.ndim != 2
            or landmarks_px.shape[0] < 2
            or landmarks_px.shape[1] < 1
        ):
            self.last_face_size = float("nan")
            self.last_face_bbox = None
            self.last_face_width = None
            self.last_face_height = None
            return float("nan")

        x_coords = landmarks_px[0].astype(float, copy=False)
        y_coords = landmarks_px[1].astype(float, copy=False)

        x_min = float(np.min(x_coords))
        x_max = float(np.max(x_coords))
        y_min = float(np.min(y_coords))
        y_max = float(np.max(y_coords))

        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)
        face_size = float(np.hypot(width, height))

        self.last_face_bbox = (x_min, y_min, x_max, y_max)
        self.last_face_width = width
        self.last_face_height = height
        self.last_face_size = face_size
        return face_size

    def _evaluate_face_direction(self, face_direction_x: float, face_direction_y: float) -> None:
        """Evaluate if face is properly oriented."""
        if self.face_state == 3:
            return
        
        if np.abs(face_direction_x) > self.face_direction_x_threshold:
            self.face_state = 2
        elif (face_direction_y < self.face_direction_y_threshold[0] or 
              face_direction_y > self.face_direction_y_threshold[1]):
            self.face_state = 2
        else:
            self.face_state = 0

    def reset(self) -> None:
        """Reset all state arrays and counters."""
        self.flag_list = np.array([0] * self.sequence_length)
        self.face_state_list = np.array([0] * self.sequence_length)
        self.munching_count = 0
        self.face_state = 0
        self.is_chewing = False
        self.last_face_size = None
        self.last_face_bbox = None
        self.last_face_width = None
        self.last_face_height = None
