"""WebSocket detection server using ASGI + python-socketio.

Streams face mesh landmarks and receives chewing detection results.
"""
from __future__ import annotations

import json
import logging
import time
from threading import Lock

import numpy as np
import socketio

from util.chewing_detector import ChewingDetector
from util.ml_predictor import load_predictor, MetricLandmarkConverter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Socket.IO server (ASGI)
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
)


async def health_app(scope, receive, send):
    if scope.get("type") != "http":
        return

    if scope.get("path") != "/":
        status = 404
        body = b"Not Found"
        headers = [(b"content-type", b"text/plain")]
    else:
        status = 200
        body = json.dumps(
            {"status": "ok", "message": "ChewingDetection WebSocket Server"}
        ).encode("utf-8")
        headers = [(b"content-type", b"application/json")]

    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


app = socketio.ASGIApp(sio, other_asgi_app=health_app)

# Limits and throttling
MAX_CLIENTS = 3
PREDICT_INTERVAL_SEC = 0.1

# Frame defaults (used when client does not provide dimensions)
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080
DEFAULT_FOCAL_LENGTH_Y = 1750.0

FOCAL_LENGTH_Y = DEFAULT_FOCAL_LENGTH_Y

# ML Prediction setup (matching main.py)
try:
    import os

    # Load ML models
    MODEL_DIR = "data/tmp_model/exp001"
    MODEL_NAMES = [
        "lgb_0.model",
        "lgb_1.model",
        "lgb_2.model",
        "lgb_3.model",
        "lgb_4.model",
    ]

    # Debug: Check if model files exist
    logger.info(f"Checking model directory: {MODEL_DIR}")
    if os.path.exists(MODEL_DIR):
        files = os.listdir(MODEL_DIR)
        logger.info(f"Files in {MODEL_DIR}: {files}")
    else:
        logger.error(f"Model directory does not exist: {MODEL_DIR}")
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    logger.info(f"ML prediction enabled with {len(MODEL_NAMES)} models")
except Exception as e:
    logger.error(f"Failed to load ML models: {e}", exc_info=True)
    raise

clients_lock = Lock()

# Face movement gating to suppress chewing detection on large jumps.
MAX_FACE_MOVE_RATIO = 0.8
_last_face_centers = {}

# Per-connection state
_active_sids: set[str] = set()
_chewing_detectors: dict[str, ChewingDetector] = {}
_ml_predictors: dict[str, object] = {}
_metric_converters: dict[str, MetricLandmarkConverter] = {}
_frame_sizes: dict[str, tuple[int, int]] = {}
_last_predict_at: dict[str, float] = {}
_last_results: dict[str, dict] = {}
_sid_locks: dict[str, Lock] = {}
_last_ml_predictions: dict[str, np.ndarray] = {}


def _coerce_frame_size(value: object, fallback: int) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return fallback
    return size if size > 0 else fallback


def _face_top_px(landmarks_px: np.ndarray) -> tuple[float, float]:
    y_values = landmarks_px[1]
    top_index = int(np.argmin(y_values))
    return (float(landmarks_px[0, top_index]), float(y_values[top_index]))


def _movement_and_suppression(
    sid: str, landmarks_px: np.ndarray, frame_width: int, frame_height: int
) -> tuple[float | None, bool]:
    center = _face_top_px(landmarks_px)
    threshold = MAX_FACE_MOVE_RATIO * float(min(frame_width, frame_height))

    with clients_lock:
        previous = _last_face_centers.get(sid)
        _last_face_centers[sid] = center

    if previous is None:
        return None, False

    dx = center[0] - previous[0]
    dy = center[1] - previous[1]
    movement = (dx * dx + dy * dy) ** 0.5
    return movement, movement > threshold


def _create_chewing_detector() -> ChewingDetector:
    return ChewingDetector(
        sequence_length=30,
        firstbite_threshold=0.26,
        mouth_gap_threshold=15.0,
        min_face_size_threshold=250.0,
        non_continuous_sounds=5,
        face_direction_x_threshold=30.0,
        face_direction_y_threshold=(-20.0, 20.0),
    )


def _init_client_state(sid: str) -> None:
    _chewing_detectors[sid] = _create_chewing_detector()
    _frame_sizes[sid] = (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
    _last_predict_at[sid] = 0.0
    _last_results[sid] = {}
    _sid_locks[sid] = Lock()
    _last_ml_predictions[sid] = None
    _ml_predictors[sid] = load_predictor(
        model_dir=MODEL_DIR,
        model_names=MODEL_NAMES,
        sequence_length=6,
        num_lag=5,
    )
    _metric_converters[sid] = MetricLandmarkConverter(
        frame_width=DEFAULT_FRAME_WIDTH,
        frame_height=DEFAULT_FRAME_HEIGHT,
        focal_length_y=FOCAL_LENGTH_Y,
    )


def _clear_client_state(sid: str) -> None:
    _active_sids.discard(sid)
    _chewing_detectors.pop(sid, None)
    _ml_predictors.pop(sid, None)
    _metric_converters.pop(sid, None)
    _frame_sizes.pop(sid, None)
    _last_predict_at.pop(sid, None)
    _last_results.pop(sid, None)
    _sid_locks.pop(sid, None)
    _last_ml_predictions.pop(sid, None)
    _last_face_centers.pop(sid, None)


@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info("Client connected: %s", environ.get("REMOTE_ADDR"))
    logger.debug("Handshake headers: %s", environ.get("headers"))
    should_reject = False
    with clients_lock:
        if len(_active_sids) >= MAX_CLIENTS:
            should_reject = True
        else:
            _active_sids.add(sid)
            _init_client_state(sid)

    if should_reject:
        logger.warning("Max clients (%d) exceeded, rejecting client: %s", MAX_CLIENTS, sid)
        await sio.emit(
            "response",
            {
                "status": "rejected",
                "message": "Max clients exceeded",
                "max_clients": MAX_CLIENTS,
            },
            to=sid,
        )
        await sio.disconnect(sid)
        return

    await sio.emit(
        "response",
        {
            "status": "connected",
            "message": "Connected to chewing detection server",
        },
        to=sid,
    )


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info("Client disconnected: %s", sid)
    with clients_lock:
        _clear_client_state(sid)


@sio.on("landmarks")
async def handle_landmarks(sid, data):
    """Handle incoming face landmarks.
    
    Expected data format:
    {
        "landmarks": [[x0, x1, ...], [y0, y1, ...], [z0, z1, ...]],  # Normalized (0-1)
        "face_direction_x": float,
        "face_direction_y": float,
        "frame_width": int,
        "frame_height": int,
    }
    """
    try:
        # Parse landmarks
        landmarks_list = data.get("landmarks")
        if not landmarks_list:
            await sio.emit("detection_result", {"error": "No landmarks provided"}, to=sid)
            return
        
        # Convert to numpy array - these are NORMALIZED landmarks (0-1 range)
        landmarks_normalized = np.array(landmarks_list, dtype=np.float32)  # (3, N)
        logger.debug("Landmarks numpy shape: %s dtype=%s", landmarks_normalized.shape, landmarks_normalized.dtype)
        
        # Get face direction
        face_direction_x = data.get("face_direction_x", 0.0)
        face_direction_y = data.get("face_direction_y", 0.0)
        logger.debug("Face direction: x=%s y=%s", face_direction_x, face_direction_y)

        with clients_lock:
            if sid not in _active_sids:
                await sio.emit("detection_result", {"error": "Unknown client"}, to=sid)
                return
            chewing_detector = _chewing_detectors[sid]
            ml_predictor = _ml_predictors.get(sid)
            metric_converter = _metric_converters.get(sid)
            frame_width, frame_height = _frame_sizes.get(
                sid, (DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
            )
            sid_lock = _sid_locks[sid]

        frame_width = _coerce_frame_size(data.get("frame_width"), frame_width)
        frame_height = _coerce_frame_size(data.get("frame_height"), frame_height)
        if (frame_width, frame_height) != _frame_sizes.get(sid, (frame_width, frame_height)):
            with clients_lock:
                _frame_sizes[sid] = (frame_width, frame_height)
                if metric_converter is not None:
                    metric_converter = MetricLandmarkConverter(
                        frame_width=frame_width,
                        frame_height=frame_height,
                        focal_length_y=FOCAL_LENGTH_Y,
                    )
                    _metric_converters[sid] = metric_converter
        
        # ML Prediction
        prediction_scores = None
        if ml_predictor is not None and metric_converter is not None:
            try:
                # Convert normalized landmarks to metric landmarks
                metric_landmarks = metric_converter.convert(landmarks_normalized)

                # Append data to predictor buffer
                current_time = time.time()
                with sid_lock:
                    ml_predictor.append_data(
                        metric_landmarks=metric_landmarks,
                        timestamp=current_time,
                        face_direction_x=face_direction_x,
                        face_direction_y=face_direction_y,
                    )

                    last_predict_at = _last_predict_at.get(sid, 0.0)
                    if current_time - last_predict_at >= PREDICT_INTERVAL_SEC:
                        prediction_scores = ml_predictor.predict()
                        _last_predict_at[sid] = current_time
                        if prediction_scores is not None:
                            _last_ml_predictions[sid] = prediction_scores
                    else:
                        prediction_scores = None
                    
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
        # (Using client frame dimensions when provided)
        landmarks_px = landmarks_normalized.copy()
        landmarks_px[0] *= frame_width
        landmarks_px[1] *= frame_height

        movement, suppress_chewing = _movement_and_suppression(
            sid, landmarks_px, frame_width, frame_height
        )
        
        # Perform detection
        if suppress_chewing:
            with sid_lock:
                flag = 0
                mouth_gap = 0.0
                face_state = chewing_detector.face_state
        else:
            with sid_lock:
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

        face_size = chewing_detector.last_face_size
        face_size_out = (
            float(face_size)
            if (face_size is not None and np.isfinite(face_size))
            else None
        )
        
        # Send result
        ml_prediction_out = None
        with clients_lock:
            cached_prediction = _last_ml_predictions.get(sid)
        if prediction_scores is not None:
            ml_prediction_out = prediction_scores.tolist()
        elif cached_prediction is not None:
            ml_prediction_out = cached_prediction.tolist()

        result = {
            "flag": int(flag),  # 0=none, 1=first_bite, 2=chewing
            "mouth_gap": float(mouth_gap),
            "face_size": face_size_out,
            "face_state": int(face_state),
            "munching_count": int(chewing_detector.munching_count),
            "movement": float(movement) if movement is not None else None,
            "ml_prediction": ml_prediction_out,
        }
        with clients_lock:
            _last_results[sid] = result
        await sio.emit("detection_result", result, to=sid)
        
    except Exception as e:
        logger.error(f"Error processing landmarks: {e}", exc_info=True)
        await sio.emit("detection_result", {"error": str(e)}, to=sid)


@sio.on("reset")
async def handle_reset(sid):
    """Reset detector state."""
    try:
        logger.info("Reset requested by client.")
        with clients_lock:
            chewing_detector = _chewing_detectors.get(sid)
            ml_predictor = _ml_predictors.get(sid)
            sid_lock = _sid_locks.get(sid)
        if sid_lock:
            with sid_lock:
                if chewing_detector is not None:
                    chewing_detector.reset()
                if ml_predictor is not None:
                    ml_predictor.reset()
        await sio.emit("detection_result", {"status": "reset"}, to=sid)
        logger.info("Detector reset (including ML predictor)")
    except Exception as e:
        logger.error(f"Error resetting detector: {e}", exc_info=True)
        await sio.emit("detection_result", {"error": str(e)}, to=sid)


@sio.on("get_state")
async def handle_get_state(sid):
    """Get current detector state."""
    try:
        logger.debug("State requested by client.")
        with clients_lock:
            chewing_detector = _chewing_detectors.get(sid)
            ml_predictor = _ml_predictors.get(sid)
            sid_lock = _sid_locks.get(sid)
        if sid_lock:
            with sid_lock:
                state = {
                    "is_chewing": chewing_detector.is_chewing if chewing_detector else False,
                    "munching_count": int(chewing_detector.munching_count) if chewing_detector else 0,
                    "face_state": int(chewing_detector.face_state) if chewing_detector else 0,
                    "ml_buffer_length": len(ml_predictor.data_buffer)
                    if ml_predictor
                    else 0,
                }
        else:
            state = {
                "is_chewing": False,
                "munching_count": 0,
                "face_state": 0,
                "ml_buffer_length": 0,
            }
        logger.debug("Current detector state: %s", state)
        await sio.emit("detector_state", state, to=sid)
    except Exception as e:
        logger.error(f"Error getting state: {e}", exc_info=True)
        await sio.emit("detector_state", {"error": str(e)}, to=sid)


if __name__ == "__main__":
    # Run server
    import uvicorn

    logger.info("Starting ChewingDetection WebSocket Server on 0.0.0.0:5000")
    logger.info("ML Prediction: ENABLED")
    logger.info("Using ASGI async mode")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
