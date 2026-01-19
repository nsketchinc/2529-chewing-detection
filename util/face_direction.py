"""Calculate face direction (yaw and pitch) from landmarks.

Responsibility: Compute face orientation angles.
"""
from __future__ import annotations

import cv2
import numpy as np


class FaceDirectionCalculator:
    """Calculate face orientation (pitch and yaw) using PnP algorithm."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        """Initialize face direction calculator.
        
        Args:
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Camera matrix
        focal_length = 1 * frame_width
        self.cam_matrix = np.array([
            [focal_length, 0, frame_height / 2],
            [0, focal_length, frame_width / 2],
            [0, 0, 1]
        ])
        
        # Distortion parameters (zero distortion)
        self.dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Key landmark indices for face direction
        # nose, left eye inner, left eye outer, chin, right eye inner, right eye outer
        self.key_indices = [1, 33, 61, 199, 263, 291]

    def calculate_direction(self, landmarks: np.ndarray) -> tuple[float, float]:
        """Calculate face direction from normalized landmarks.
        
        Args:
            landmarks: Normalized landmarks (3, 468) with x, y, z
            
        Returns:
            face_direction_y: Pitch angle (up/down) in degrees
            face_direction_x: Yaw angle (left/right) in degrees
        """
        # Extract key points
        face_3d = landmarks[:, self.key_indices].T  # (6, 3)
        
        # Scale to pixel coordinates
        face_3d[:, 0] = (face_3d[:, 0] * self.frame_width).astype(int)
        face_3d[:, 1] = (face_3d[:, 1] * self.frame_height).astype(int)
        # z coordinate stays normalized
        
        # Solve PnP
        face_2d = face_3d[:, :2].astype(np.float32)
        success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, self.cam_matrix, self.dist_matrix)
        
        if not success:
            return 0.0, 0.0
        
        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)
        
        # Get angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Extract pitch and yaw
        face_direction_y = angles[0] * 360  # Pitch (up/down)
        face_direction_x = angles[1] * 360  # Yaw (left/right)
        
        return face_direction_y, face_direction_x
