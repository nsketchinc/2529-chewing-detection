# ML Integration Guide

## Overview

The ML prediction feature allows the chewing detection system to use trained machine learning models for more accurate detection.

## Setup

### 1. Prepare Model Files

Place your trained model pickle files in a directory (e.g., `data/`):
```
data/
  ├── model1.pkl
  ├── model2.pkl
  └── model3.pkl
```

### 2. Prepare Training Module (Optional)

If you have a custom `training.py` module with `get_preprocess()` and `get_lag_features()` functions, place it in the project root. Otherwise, the default stub implementation will be used.

### 3. Configure main.py

Edit `main.py` to enable ML prediction:

```python
def main():
    use_ml = True  # Enable ML prediction
    model_dir = "data"  # Your model directory
    model_names = ["model1.pkl", "model2.pkl", "model3.pkl"]  # Your model files
    
    app = ChewingDetectionApp(
        camera_index=None,
        use_ml_prediction=use_ml,
        model_dir=model_dir if use_ml else None,
        model_names=model_names if use_ml else None,
    )
    app.run()
```

## How It Works

### Data Flow

1. **Face Detection**: MediaPipe detects 468 facial landmarks (normalized coordinates)
2. **Metric Conversion**: Landmarks are converted to metric (3D) coordinates using `MetricLandmarkConverter`
3. **Data Buffering**: Last 6 frames of metric landmarks, timestamps, and face directions are stored
4. **Feature Generation**: Raw data is preprocessed and lag features are generated
5. **Prediction**: All loaded models predict and results are averaged
6. **Detection Logic**: Predictions are passed to `ChewingDetector` which uses thresholds to trigger events

### Prediction Output

Models should output a 3-element array:
- `[0]`: Other/no action score
- `[1]`: First bite score
- `[2]`: Onwards bite score

### Thresholds

Configure thresholds in `ChewingDetector` initialization:
```python
self.chewing_detector = ChewingDetector(
    sequence_length=30,
    firstbite_threshold=-0.5,  # Threshold for first bite
    onwards_threshold=-0.3,     # Threshold for onwards bite
)
```

## Architecture

### New Files

- **`util/ml_predictor.py`**: 
  - `MLPredictor`: Loads models, buffers data, generates features, runs predictions
  - `MetricLandmarkConverter`: Converts normalized landmarks to metric coordinates (mimics face_geometry PCF)
  - `load_predictor()`: Convenience function for initialization

- **`training.py`**: 
  - `get_preprocess()`: Feature preprocessing (stub provided)
  - `get_lag_features()`: Time-series lag feature generation (stub provided)

### Integration Points

In `main.py`:
1. **Initialization**: ML predictor and metric converter are created if enabled
2. **Detection Loop**: 
   - Metric landmarks are calculated
   - Data is appended to buffer
   - Prediction is obtained
   - Prediction scores are passed to `ChewingDetector`
3. **Visualization**: ML scores are displayed on screen (if available)

## Without ML

The system works without ML prediction. In this case:
- Set `use_ml_prediction=False` in main()
- `ChewingDetector` will use only mouth movement analysis (`first_tojihantei`, `onwards_tojihantei`)
- Detection will be based purely on geometric analysis

## Customization

### Custom Preprocessing

Replace `training.py` with your actual preprocessing pipeline:

```python
def get_preprocess(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    # Your preprocessing logic
    processed = your_scaler.transform(data)
    return processed, feature_names

def get_lag_features(data, feat_cols, num_lag=5):
    # Your lag feature generation
    return lagged_data, lagged_columns
```

### Custom Metric Conversion

If you need the actual face_geometry library:

```python
# In util/ml_predictor.py, replace MetricLandmarkConverter with:
from face_geometry import get_metric_landmarks, PCF

pcf = PCF(near=1, far=10000, frame_height=height, frame_width=width, fy=1750)
metric_landmarks, _ = get_metric_landmarks(normalized_landmarks, pcf)
```

## Troubleshooting

### ImportError: training module not found
- Create `training.py` with your preprocessing functions
- Or use the provided stub (less accurate but works)

### Model prediction errors
- Ensure models expect the correct feature shape
- Check that preprocessing matches training pipeline
- Verify model files are not corrupted

### No detection even with ML
- Check threshold values (make them less strict by moving toward 0)
- Verify ML scores are being calculated (check console output)
- Ensure face is properly oriented (not turned away)

## Example: Complete Configuration

```python
# main.py
def main():
    try:
        # ML Configuration
        use_ml = True
        model_dir = "path/to/models"
        model_names = ["lgbm_model_fold0.pkl", "lgbm_model_fold1.pkl"]
        
        app = ChewingDetectionApp(
            camera_index=None,
            use_ml_prediction=use_ml,
            model_dir=model_dir if use_ml else None,
            model_names=model_names if use_ml else None,
        )
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
```

## Reference

This implementation is based on the original `mediapipe_soshaku.py` with:
- Modularized architecture using SOLID principles
- Separated concerns (detection, prediction, visualization)
- Optional ML integration
- Fallback to geometric analysis when ML is disabled
