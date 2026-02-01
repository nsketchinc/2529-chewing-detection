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
        non_continuous_sounds: int = 5,
        face_direction_x_threshold: float = 30.0,
        face_direction_y_threshold: tuple[float, float] = (-20.0, 20.0),
    ) -> None:
        """Initialize chewing detector with configurable thresholds.
        
        Args:
            sequence_length: Number of frames to keep in history
            firstbite_threshold: Threshold for first bite detection
            mouth_gap_threshold: Max mouth opening allowed to count as chewing
            non_continuous_sounds: Minimum frames between sound triggers
            face_direction_x_threshold: Max horizontal face angle deviation
            face_direction_y_threshold: Min/max vertical face angle range
        """
        self.sequence_length = sequence_length
        self.firstbite_threshold = firstbite_threshold
        self.mouth_gap_threshold = mouth_gap_threshold
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

    def detect_chewing(
        self,
        landmarks_px: np.ndarray,
        face_direction_x: float,
        face_direction_y: float,
        prediction_scores: np.ndarray | None = None,
    ) -> tuple[int, float, int]:
        """Detect chewing action from current frame.
        
        Args:
            landmarks_px: Pixel coordinates (2, 468)
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
        
        # Evaluate face direction
        self._evaluate_face_direction(face_direction_x, face_direction_y)

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
