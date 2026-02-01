"""Machine learning predictor for chewing detection.

Responsibility: Load trained models and predict chewing events from landmark time series.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MLPredictor:
    """Predict chewing events using trained ML models."""

    def __init__(
        self,
        model_paths: list[str | Path],
        sequence_length: int = 6,
        num_lag: int = 5,
        model_dir: str | Path | None = None,
    ) -> None:
        """Initialize ML predictor with trained models.
        
        Args:
            model_paths: List of paths to pickled model files
            sequence_length: Number of frames to keep for prediction
            num_lag: Number of lag features to generate
            model_dir: Directory containing feat_cols.pickle (optional)
        """
        self.models = []
        self.sequence_length = sequence_length
        self.num_lag = num_lag
        self.data_buffer = []
        self.feat_cols = None
        
        # Load models
        for model_path in model_paths:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(path, 'rb') as f:
                model = pickle.load(f)
                self.models.append(model)
        
        print(f"Loaded {len(self.models)} ML models")
        
        # Try to load feat_cols.pickle
        if model_dir is not None:
            feat_cols_path = Path(model_dir) / "feat_cols.pickle"
            if feat_cols_path.exists():
                try:
                    with open(feat_cols_path, 'rb') as f:
                        self.feat_cols = pickle.load(f)
                    print(f"Loaded feature columns from {feat_cols_path}: {len(self.feat_cols)} features")
                except Exception as e:
                    print(f"Warning: Failed to load feat_cols.pickle: {e}")
            else:
                print(f"Warning: feat_cols.pickle not found at {feat_cols_path}")

    def append_data(
        self,
        metric_landmarks: np.ndarray,
        timestamp: float,
        face_direction_x: float,
        face_direction_y: float,
    ) -> None:
        """Append new frame data to the buffer.
        
        Args:
            metric_landmarks: Metric landmarks (3, 468) in meters
            timestamp: Frame timestamp
            face_direction_x: Horizontal face angle
            face_direction_y: Vertical face angle
        """
        self.data_buffer.append({
            'landmarks': metric_landmarks,
            'timestamp': timestamp,
            'direction_x': face_direction_x,
            'direction_y': face_direction_y,
        })
        
        # Keep only recent data (plus some margin)
        max_buffer_size = 100
        if len(self.data_buffer) > max_buffer_size:
            self.data_buffer = self.data_buffer[-max_buffer_size:]

    def predict(self) -> np.ndarray | None:
        """Predict chewing event from buffered data.
        
        Returns:
            Prediction array [first_bite_score, onwards_score, other_score]
            or None if insufficient data
        """
        if len(self.data_buffer) < self.sequence_length:
            return None
        
        # Extract recent frames
        recent_data = self.data_buffer[-self.sequence_length:]
        
        # Stack landmarks
        landmarks = np.stack([d['landmarks'] for d in recent_data])  # (seq_len, 3, 468)
        timestamps = np.array([d['timestamp'] for d in recent_data])
        direction_x = np.array([d['direction_x'] for d in recent_data])
        direction_y = np.array([d['direction_y'] for d in recent_data])
        
        # Create feature array
        features = self._create_features(landmarks, timestamps, direction_x, direction_y)
        
        # Predict with all models and average
        features_last = features[-1:, :]
        
        # If we have feature column names, create a DataFrame
        if self.feat_cols is not None and len(self.feat_cols) == features_last.shape[1]:
            features_last = pd.DataFrame(features_last, columns=self.feat_cols)

        predictions = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(features_last)[0]
            else:
                pred = model.predict(features_last)[0]
            predictions.append(pred)
        
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction

    def _create_features(
        self,
        landmarks: np.ndarray,
        timestamps: np.ndarray,
        direction_x: np.ndarray,
        direction_y: np.ndarray,
    ) -> np.ndarray:
        """Create feature array from raw data.
        
        Args:
            landmarks: (seq_len, 3, 468)
            timestamps: (seq_len,)
            direction_x: (seq_len,)
            direction_y: (seq_len,)
            
        Returns:
            Feature array (seq_len, num_features) - expects 258 features with num_lag=5
            This corresponds to 21 landmarks * 2 (x,y) + 1 (time) = 43 base features
            With 5 lag steps: 43 * 6 = 258 features
        """
        # Select key landmarks for mouth/chewing (21 landmarks):
        # MediaPipe face landmarks for mouth region:
        # - 61-68: Outer lips top (8 points)
        # - 71-78: Outer lips bottom (8 points)  
        # - 0: Face center/reference (1 point)
        # - 164, 165: Key mouth interior points (2 points)
        # - 10, 11: Chin area (2 points)
        mouth_landmarks = [0, 10, 11, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 164, 165]
        
        # Extract selected landmarks: x and y coordinates
        x = landmarks[:, 0, mouth_landmarks]  # (seq_len, 21)
        y = landmarks[:, 1, mouth_landmarks]  # (seq_len, 21)
        
        # Time differences
        times = np.diff(timestamps, prepend=1e5)  # (seq_len,)
        
        # Concatenate selected features: 21*2 + 1 = 43 base features
        data = np.concatenate(
            [x, y, times[:, None]],
            axis=1
        )  # (seq_len, 43 features before lag)
        
        # Apply preprocessing (if training module is available)
        try:
            from training import get_preprocess, get_lag_features
            data, feat_cols = get_preprocess(data)
            data, temp_feat_cols = get_preprocess(data)
            data, self.feat_cols = get_lag_features(data, temp_feat_cols, self.num_lag)
            print(f"Using training module preprocessing: {len(self.feat_cols)} features")
        except ImportError:
            # If training module not available, apply lag features manually
            print("Warning: training module not found, applying lag features manually")
            
            # Simple lag feature generation (43 base features * (1 + num_lag) lags)
            features_with_lag = []
            for i in range(len(data)):
                frame_features = []
                for lag in range(self.num_lag + 1):
                    idx = max(0, i - lag)
                    frame_features.append(data[idx])
                features_with_lag.append(np.concatenate(frame_features))
            
            data = np.array(features_with_lag)  # (seq_len, 43 * (num_lag + 1))
            
            # If feat_cols was loaded from pickle, verify shape matches
            if self.feat_cols is not None and len(self.feat_cols) != data.shape[1]:
                print(f"Warning: feat_cols length ({len(self.feat_cols)}) doesn't match features ({data.shape[1]})")

        return data

    def reset(self) -> None:
        """Clear the data buffer."""
        self.data_buffer = []
        self.feat_cols = None


class MetricLandmarkConverter:
    """Convert normalized landmarks to metric (3D) landmarks.

    - Tries to use face_geometry (PCF + get_metric_landmarks) when available
    - Falls back to a simplified projection if the library is missing
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        focal_length_y: float = 1750.0,
        near: float = 1.0,
        far: float = 10000.0,
    ) -> None:
        """Initialize metric landmark converter."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length_y = focal_length_y
        self.near = near
        self.far = far

        # Try to load the actual face_geometry implementation
        try:
            from util.face_geometry import PCF, get_metric_landmarks  # type: ignore

            self._pcf = PCF(
                near=self.near,
                far=self.far,
                frame_height=self.frame_height,
                frame_width=self.frame_width,
                fy=self.focal_length_y,
            )
            self._get_metric_landmarks = get_metric_landmarks
            self._use_face_geometry = True
        except ImportError:
            # Fallback if face_geometry is not installed/on path
            self._pcf = None
            self._get_metric_landmarks = None
            self._use_face_geometry = False

    def convert(self, normalized_landmarks: np.ndarray) -> np.ndarray:
        """Convert normalized landmarks to metric coordinates."""
        # FaceGeometryは468点前提。最新FaceLandmarkerは478点返すので先頭468に揃える
        if normalized_landmarks.shape[1] > 468:
            normalized_landmarks = normalized_landmarks[:, :468]

        if self._use_face_geometry and self._pcf is not None and self._get_metric_landmarks is not None:
            # Use the official PCF-based conversion
            metric_landmarks, _ = self._get_metric_landmarks(normalized_landmarks.copy(), self._pcf)
            return metric_landmarks

        # Fallback: simplified conversion (approximation)
        metric_landmarks = np.zeros_like(normalized_landmarks)
        metric_landmarks[0] = (normalized_landmarks[0] - 0.5) * self.frame_width / self.focal_length_y
        metric_landmarks[1] = (normalized_landmarks[1] - 0.5) * self.frame_height / self.focal_length_y
        metric_landmarks[2] = normalized_landmarks[2] * 100.0  # Rough scaling
        return metric_landmarks


def load_predictor(
    model_dir: str | Path,
    model_names: list[str],
    sequence_length: int = 6,
    num_lag: int = 5,
) -> MLPredictor:
    """Convenience function to load predictor from directory.
    
    Args:
        model_dir: Directory containing model files
        model_names: List of model filenames
        sequence_length: Number of frames for prediction
        num_lag: Number of lag features
        
    Returns:
        Initialized MLPredictor
    """
    model_dir = Path(model_dir)
    model_paths = [model_dir / name for name in model_names]
    return MLPredictor(model_paths, sequence_length, num_lag)
