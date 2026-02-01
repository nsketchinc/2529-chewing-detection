"""WebSocket detection server using Flask-SocketIO.

Streams face mesh landmarks and receives chewing detection results.
"""
from __future__ import annotations

import json
import logging
import time
from threading import Lock

import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit

from util.chewing_detector import ChewingDetector
from util.ml_predictor import load_predictor, MetricLandmarkConverter


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "chewing-detection-secret"

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    async_mode="threading",
)

# Global state
chewing_detector = ChewingDetector(
    sequence_length=30,
    firstbite_threshold=0.15,
    mouth_gap_threshold=15.0,
    non_continuous_sounds=5,
    face_direction_x_threshold=30.0,
    face_direction_y_threshold=(-20.0, 20.0),
)

# ML Prediction setup (matching main.py)
USE_ML_PREDICTION = True
ml_predictor = None
metric_converter = None

if USE_ML_PREDICTION:
    try:
        # Frame dimensions (default to 1920x1080, matching frontend ideal constraints)
        FRAME_WIDTH = 1920
        FRAME_HEIGHT = 1080
        FOCAL_LENGTH_Y = 1750.0
        
        # Load ML models
        MODEL_DIR = "data/tmp_model/exp001"
        MODEL_NAMES = [
            "lgb_0.model",
            "lgb_1.model",
            "lgb_2.model",
            "lgb_3.model",
            "lgb_4.model",
        ]
        
        ml_predictor = load_predictor(
            model_dir=MODEL_DIR,
            model_names=MODEL_NAMES,
            sequence_length=6,
            num_lag=5,
        )
        
        metric_converter = MetricLandmarkConverter(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            focal_length_y=FOCAL_LENGTH_Y,
        )
        
        logger.info(f"ML prediction enabled with {len(MODEL_NAMES)} models")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}", exc_info=True)
        USE_ML_PREDICTION = False
        ml_predictor = None
        metric_converter = None

detector_lock = Lock()


@app.route("/")
def index():
    """Health check endpoint."""
    return {"status": "ok", "message": "ChewingDetection WebSocket Server"}


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {socketio.server.environ.get('REMOTE_ADDR')}")
    logger.debug("Handshake headers: %s", socketio.server.environ.get("headers"))
    emit("response", {
        "status": "connected", 
        "message": "Connected to chewing detection server",
        "ml_enabled": USE_ML_PREDICTION
    })


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {socketio.server.environ.get('REMOTE_ADDR')}")


@socketio.on("landmarks")
def handle_landmarks(data):
    """Handle incoming face landmarks.
    
    Expected data format:
    {
        "landmarks": [[x0, x1, ...], [y0, y1, ...], [z0, z1, ...]],  # Normalized (0-1)
        "face_direction_x": float,
        "face_direction_y": float,
    }
    """
    try:
        # Parse landmarks
        landmarks_list = data.get("landmarks")
        if not landmarks_list:
            emit("detection_result", {"error": "No landmarks provided"})
            return
        
        # Convert to numpy array - these are NORMALIZED landmarks (0-1 range)
        landmarks_normalized = np.array(landmarks_list, dtype=np.float32)  # (3, N)
        logger.debug("Landmarks numpy shape: %s dtype=%s", landmarks_normalized.shape, landmarks_normalized.dtype)
        
        # Get face direction
        face_direction_x = data.get("face_direction_x", 0.0)
        face_direction_y = data.get("face_direction_y", 0.0)
        logger.debug("Face direction: x=%s y=%s", face_direction_x, face_direction_y)
        
        # ML Prediction (if enabled)
        prediction_scores = None
        if USE_ML_PREDICTION and ml_predictor is not None and metric_converter is not None:
            try:
                # Convert normalized landmarks to metric landmarks
                metric_landmarks = metric_converter.convert(landmarks_normalized)
                
                # Append data to predictor buffer
                current_time = time.time()
                with detector_lock:
                    ml_predictor.append_data(
                        metric_landmarks=metric_landmarks,
                        timestamp=current_time,
                        face_direction_x=face_direction_x,
                        face_direction_y=face_direction_y,
                    )
                    
                    # Get prediction
                    prediction_scores = ml_predictor.predict()
                    
                if prediction_scores is not None:
                    logger.debug(
                        "ML prediction: [%.3f, %.3f, %.3f]",
                        prediction_scores[0],
                        prediction_scores[1],
                        prediction_scores[2],
                    )
                else:
                    logger.debug("ML prediction: buffering (%d/6)", len(ml_predictor.data_buffer))
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}", exc_info=True)
                prediction_scores = None
        
        # Convert normalized landmarks to pixel coordinates for chewing detector
        # (Using default frame dimensions)
        landmarks_px = landmarks_normalized.copy()
        landmarks_px[0] *= FRAME_WIDTH if USE_ML_PREDICTION else 1920
        landmarks_px[1] *= FRAME_HEIGHT if USE_ML_PREDICTION else 1080
        
        # Perform detection
        with detector_lock:
            flag, mouth_gap, face_state = chewing_detector.detect_chewing(
                landmarks_px=landmarks_px,
                face_direction_x=face_direction_x,
                face_direction_y=face_direction_y,
                prediction_scores=prediction_scores,
            )
        logger.debug(
            "Detection result raw: flag=%s mouth_gap=%s face_state=%s munching_count=%s",
            flag,
            mouth_gap,
            face_state,
            chewing_detector.munching_count,
        )
        
        # Send result
        result = {
            "flag": int(flag),  # 0=none, 1=first_bite, 2=chewing
            "mouth_gap": float(mouth_gap),
            "face_state": int(face_state),
            "munching_count": int(chewing_detector.munching_count),
            "ml_prediction": prediction_scores.tolist() if prediction_scores is not None else None,
        }
        emit("detection_result", result)
        
    except Exception as e:
        logger.error(f"Error processing landmarks: {e}", exc_info=True)
        emit("detection_result", {"error": str(e)})


@socketio.on("reset")
def handle_reset():
    """Reset detector state."""
    try:
        logger.info("Reset requested by client.")
        with detector_lock:
            chewing_detector.reset()
            if USE_ML_PREDICTION and ml_predictor is not None:
                ml_predictor.reset()
        emit("detection_result", {"status": "reset"})
        logger.info("Detector reset (including ML predictor)" if USE_ML_PREDICTION else "Detector reset")
    except Exception as e:
        logger.error(f"Error resetting detector: {e}", exc_info=True)
        emit("detection_result", {"error": str(e)})


@socketio.on("get_state")
def handle_get_state():
    """Get current detector state."""
    try:
        logger.debug("State requested by client.")
        with detector_lock:
            state = {
                "is_chewing": chewing_detector.is_chewing,
                "munching_count": int(chewing_detector.munching_count),
                "face_state": int(chewing_detector.face_state),
                "ml_enabled": USE_ML_PREDICTION,
                "ml_buffer_length": len(ml_predictor.data_buffer) if (USE_ML_PREDICTION and ml_predictor) else 0,
            }
        logger.debug("Current detector state: %s", state)
        emit("detector_state", state)
    except Exception as e:
        logger.error(f"Error getting state: {e}", exc_info=True)
        emit("detector_state", {"error": str(e)})


if __name__ == "__main__":
    # Run server
    logger.info("Starting ChewingDetection WebSocket Server on 0.0.0.0:5000")
    logger.info(f"ML Prediction: {'ENABLED' if USE_ML_PREDICTION else 'DISABLED'}")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
