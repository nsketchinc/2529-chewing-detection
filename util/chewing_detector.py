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
        firstbite_threshold: float = -0.3,
        onwards_threshold: float = -0.15,
        first_check_threshold: float = -0.7,
        onwards_check_threshold: float = -0.3,
        non_continuous_sounds: int = 5,
        face_direction_x_threshold: float = 30.0,
        face_direction_y_threshold: tuple[float, float] = (-20.0, 20.0),
    ) -> None:
        """Initialize chewing detector with configurable thresholds.
        
        Args:
            sequence_length: Number of frames to keep in history
            firstbite_threshold: Threshold for first bite detection
            onwards_threshold: Threshold for subsequent bite detection
            first_check_threshold: Threshold for jaw closing check (first bite)
            onwards_check_threshold: Threshold for jaw closing check (onwards)
            non_continuous_sounds: Minimum frames between sound triggers
            face_direction_x_threshold: Max horizontal face angle deviation
            face_direction_y_threshold: Min/max vertical face angle range
        """
        self.sequence_length = sequence_length
        self.firstbite_threshold = firstbite_threshold
        self.onwards_threshold = onwards_threshold
        self.first_check_threshold = first_check_threshold
        self.onwards_check_threshold = onwards_check_threshold
        self.non_continuous_sounds = non_continuous_sounds
        self.face_direction_x_threshold = face_direction_x_threshold
        self.face_direction_y_threshold = face_direction_y_threshold

        # State arrays
        self.distance_array = [0.0] * sequence_length
        self.firstbite_flag_array = np.array([False] * sequence_length)
        self.onwards_flag_array = np.array([False] * sequence_length)
        self.first_tojihantei_array = np.array([False] * sequence_length)
        self.flag_list = np.array([0] * sequence_length)
        self.face_state_list = np.array([0] * sequence_length)
        
        # Counters
        self.munching_count = 0
        self.number_of_frames_since_last_onwards = 0
        self.face_state = 0
        self.first_bite_check = 0.0
        self.second_bite_check = np.array([0.0])

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
            flag: 0=none, 1=first_bite, 2/3=onwards_bite
            mouth_gap: Current mouth opening distance
            face_state: 0=ok, 1=no_face, 2=face_turned, 3=other
        """
        # Calculate mouth/jaw distances
        mouth_gap, first_tojihantei, onwards_tojihantei = self._check_mouth_closing(landmarks_px)
        
        # Update state arrays
        self._update_states(first_tojihantei)
        
        # Update prediction flags if ML scores provided
        if prediction_scores is not None:
            self._update_prediction_flags(prediction_scores)
        
        # Evaluate face direction
        self._evaluate_face_direction(face_direction_x, face_direction_y)
        
        # Determine chewing flag
        flag = self._evaluate_flag(onwards_tojihantei)
        
        return flag, mouth_gap, self.face_state

    def _check_mouth_closing(self, landmarks_px: np.ndarray) -> tuple[float, bool, bool]:
        """Check if jaw is closing (simplified bite check)."""
        # Landmark indices for mouth measurements
        mouth_above_idx = 12
        mouth_under_idx = 15
        mouth_left_idx = 78
        mouth_right_idx = 308
        
        y_coords = landmarks_px[1]
        x_coords = landmarks_px[0]
        
        # Vertical mouth gap
        distance_upper_under = float(np.abs(y_coords[mouth_above_idx] - y_coords[mouth_under_idx]))
        
        # Horizontal mouth width
        distance_left_right = float(np.abs(x_coords[mouth_left_idx] - x_coords[mouth_right_idx]))
        
        # Combined metric (gap - width)
        combined_distance = distance_upper_under - distance_left_right
        
        # Update distance array
        self.distance_array[:-1] = self.distance_array[1:]
        self.distance_array[-1] = combined_distance
        
        # Calculate rate of change
        distance_array_diff = np.diff(np.array(self.distance_array[-5:]))
        
        self.first_bite_check = float(np.sum(distance_array_diff[-2:]))
        self.second_bite_check = distance_array_diff[-2:]
        
        # Determine closing state
        first_tojihantei = self.first_bite_check < self.first_check_threshold
        onwards_tojihantei = np.any(self.second_bite_check < self.onwards_check_threshold)
        
        return distance_upper_under, first_tojihantei, onwards_tojihantei

    def _update_states(self, first_tojihantei: bool) -> None:
        """Update internal state arrays."""
        self.first_tojihantei_array[:-1] = self.first_tojihantei_array[1:]
        self.first_tojihantei_array[-1] = first_tojihantei
        
        self.flag_list[:-1] = self.flag_list[1:]

    def _update_prediction_flags(self, prediction_scores: np.ndarray) -> None:
        """Update prediction flags based on ML model output.
        
        Args:
            prediction_scores: [first_bite_score, onwards_score, other_score]
        """
        self.firstbite_flag_array[:-1] = self.firstbite_flag_array[1:]
        self.onwards_flag_array[:-1] = self.onwards_flag_array[1:]
        
        # Check if predictions exceed thresholds
        first_bite_detected = prediction_scores[0] > self.firstbite_threshold
        onwards_detected = prediction_scores[1] > self.onwards_threshold
        
        self.firstbite_flag_array[-1] = first_bite_detected
        self.onwards_flag_array[-1] = onwards_detected

    def _evaluate_face_direction(self, face_direction_x: float, face_direction_y: float) -> None:
        """Evaluate if face is properly oriented."""
        if self.face_state == 3:
            return
        
        if np.abs(face_direction_x) > self.face_direction_x_threshold:
            self.face_state = 2
        elif (face_direction_y < self.face_direction_y_threshold[0] or 
              face_direction_y > self.face_direction_y_threshold[1]):
            self.face_state = 2

    def _evaluate_flag(self, onwards_tojihantei: bool) -> int:
        """Determine chewing flag based on current state."""
        flag = 0
        
        # Face not properly oriented or recent sound
        if self.face_state in [1, 2]:
            self.number_of_frames_since_last_onwards += 1
            flag = 0
        elif np.any(self.flag_list[-self.non_continuous_sounds:] > 0):
            # Recent sound detected, prevent continuous triggering
            self.number_of_frames_since_last_onwards += 1
            flag = 0
        else:
            # Check for first bite
            if (np.any(self.firstbite_flag_array[-13:]) and 
                np.any(self.first_tojihantei_array[-1:])):
                flag = 1
                self.munching_count += 1
                self.number_of_frames_since_last_onwards = 0
            # Check for onwards bite
            elif (not np.any(self.firstbite_flag_array[-25:]) and 
                  np.all(self.onwards_flag_array[-1:]) and 
                  onwards_tojihantei):
                flag = 2
                self.number_of_frames_since_last_onwards = 0
            else:
                self.number_of_frames_since_last_onwards += 1
        
        self.flag_list[-1] = flag
        
        # Update face state history
        self.face_state_list[:-1] = self.face_state_list[1:]
        self.face_state_list[-1] = self.face_state
        
        return flag

    def reset(self) -> None:
        """Reset all state arrays and counters."""
        self.distance_array = [0.0] * self.sequence_length
        self.firstbite_flag_array = np.array([False] * self.sequence_length)
        self.onwards_flag_array = np.array([False] * self.sequence_length)
        self.first_tojihantei_array = np.array([False] * self.sequence_length)
        self.flag_list = np.array([0] * self.sequence_length)
        self.face_state_list = np.array([0] * self.sequence_length)
        self.munching_count = 0
        self.number_of_frames_since_last_onwards = 0
        self.face_state = 0
