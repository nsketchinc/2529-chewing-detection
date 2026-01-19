#!/usr/bin/env python
"""Test the ML predictor with correct feature dimensions."""
import numpy as np
from util.ml_predictor import MLPredictor

# Create test data
test_landmarks = np.random.randn(6, 3, 468)  # 6 frames, 3D, 468 points
test_timestamps = np.array([1.0, 1.033, 1.066, 1.099, 1.132, 1.165])
test_direction_x = np.zeros(6)
test_direction_y = np.zeros(6)

# Load predictor
predictor = MLPredictor(
    model_paths=['data/model/009/lgb_1.model'],
    sequence_length=6,
    num_lag=5,
)

# Append data and test prediction
for i in range(6):
    predictor.append_data(
        test_landmarks[i],
        test_timestamps[i],
        test_direction_x[i],
        test_direction_y[i],
    )

# Try prediction
try:
    result = predictor.predict()
    print(f"Prediction successful!")
    print(f"Prediction scores: {result}")
except Exception as e:
    print(f"Prediction failed: {e}")
    import traceback
    traceback.print_exc()
