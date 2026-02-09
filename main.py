"""Chewing detection application with MediaPipe and optional ML prediction.

This application integrates:
- MediaPipe face landmark detection
- Face direction calculation
- Chewing pattern detection
- Optional machine learning prediction
- Real-time visualization
"""

import time

import cv2
import numpy as np

from util.camera_config import load_camera_index
from util.chewing_detector import ChewingDetector
from util.color import Colors
from util.face_direction import FaceDirectionCalculator
from util.landmark_metrics import LandmarkMetrics
from util.mediapipe_runner import MediapipeRunner
from util.ml_predictor import MetricLandmarkConverter, load_predictor


class ChewingDetectionApp:
    """Main application for real-time chewing detection."""

    def __init__(
        self,
        camera_index: int | None = None,
        use_ml_prediction: bool = False,
        model_dir: str | None = None,
        model_names: list[str] | None = None,
    ) -> None:
        """Initialize chewing detection application.

        Args:
            camera_index: Camera device index (None = use saved config)
            use_ml_prediction: Whether to use ML model for prediction
            model_dir: Directory containing trained ML model files
            model_names: List of model filenames to load
        """
        # Camera setup
        self.camera_index = (
            camera_index if camera_index is not None else load_camera_index()
        )
        self.v_cap = cv2.VideoCapture(self.camera_index)

        if not self.v_cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.camera_index}")

        # Get frame dimensions
        self.frame_width = int(self.v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.v_cap.get(cv2.CAP_PROP_FPS)

        # Window canvas size
        self.canvas_width = 1200
        self.canvas_height = 800

        # Calculate offset to center camera image
        self.offset_x = (self.canvas_width - self.frame_width) // 2
        self.offset_y = (self.canvas_height - self.frame_height) // 2

        print(
            f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
        )
        print(
            f"Window size: {self.canvas_width}x{self.canvas_height}, offset: ({self.offset_x}, {self.offset_y})"
        )

        # Initialize components
        self.metrics = LandmarkMetrics()
        self.chewing_detector = ChewingDetector(
            sequence_length=30,
            firstbite_threshold=0.1,
            mouth_gap_threshold=15.0,
            min_face_size_threshold=250.0,
        )
        self.face_direction_calc = FaceDirectionCalculator(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )

        # ML prediction (optional)
        self.use_ml_prediction = use_ml_prediction
        self.ml_predictor = None
        self.metric_converter = None

        if use_ml_prediction:
            if not model_dir or not model_names:
                print("Warning: ML prediction enabled but no models specified")
                self.use_ml_prediction = False
            else:
                try:
                    self.ml_predictor = load_predictor(
                        model_dir=model_dir,
                        model_names=model_names,
                        sequence_length=6,
                        num_lag=5,
                    )
                    self.metric_converter = MetricLandmarkConverter(
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                        focal_length_y=1750.0,
                    )
                    print(f"ML prediction enabled with {len(model_names)} models")
                except Exception as e:
                    print(f"Failed to load ML models: {e}")
                    self.use_ml_prediction = False

        # Statistics
        self.frame_count = 0
        self.start_time = time.time()

    def run(self) -> None:
        """Run the main detection loop."""
        print("Starting chewing detection...")
        print("Press ESC to exit")

        with MediapipeRunner() as mp_runner:
            while self.v_cap.isOpened():
                success, image = self.v_cap.read()
                if not success:
                    print("Failed to read frame")
                    break

                # Create canvas and place camera image in center
                canvas = np.zeros(
                    (self.canvas_height, self.canvas_width, 3), dtype=np.uint8
                )
                canvas[
                    self.offset_y : self.offset_y + self.frame_height,
                    self.offset_x : self.offset_x + self.frame_width,
                ] = image

                # Convert to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect face landmarks
                detection = mp_runner.detect(rgb_image)

                if not detection.has_landmarks():
                    cv2.putText(
                        canvas,
                        "No face detected",
                        (self.offset_x + 10, self.offset_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        Colors.RED,
                        2,
                    )
                    cv2.imshow("Chewing Detection", canvas)

                    if cv2.waitKey(5) & 0xFF == 27:  # ESC
                        break
                    continue

                # Get landmarks
                normalized_landmarks = detection.get_normalized_landmarks()
                pixel_landmarks = detection.get_pixel_landmarks()

                # Calculate face direction
                face_direction_y, face_direction_x = (
                    self.face_direction_calc.calculate_direction(normalized_landmarks)
                )

                # ML prediction (if enabled)
                prediction_scores = None
                if self.use_ml_prediction and self.ml_predictor is not None:
                    # Convert to metric landmarks
                    metric_landmarks = self.metric_converter.convert(
                        normalized_landmarks
                    )

                    # Append data to predictor buffer
                    current_time = time.time()
                    self.ml_predictor.append_data(
                        metric_landmarks=metric_landmarks,
                        timestamp=current_time,
                        face_direction_x=face_direction_x,
                        face_direction_y=face_direction_y,
                    )

                    # Get prediction
                    prediction_scores = self.ml_predictor.predict()

                # Detect chewing
                chewing_flag, mouth_gap, face_state = (
                    self.chewing_detector.detect_chewing(
                        pixel_landmarks,
                        face_direction_x,
                        face_direction_y,
                        prediction_scores,
                    )
                )

                # Visualize results
                self._draw_visualization(
                    canvas,
                    pixel_landmarks,
                    chewing_flag,
                    mouth_gap,
                    face_state,
                    face_direction_x,
                    face_direction_y,
                    prediction_scores,
                    len(self.ml_predictor.data_buffer)
                    if (self.use_ml_prediction and self.ml_predictor is not None)
                    else 0,
                )

                # Display
                cv2.imshow("Chewing Detection", canvas)

                self.frame_count += 1

                if cv2.waitKey(5) & 0xFF == 27:  # ESC
                    print("Exiting...")
                    break

        # Cleanup
        self.v_cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\nSession statistics:")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Total chews detected: {self.chewing_detector.munching_count}")

    def _draw_visualization(
        self,
        image: np.ndarray,
        pixel_landmarks: np.ndarray,
        chewing_flag: int,
        mouth_gap: float,
        face_state: int,
        face_direction_x: float,
        face_direction_y: float,
        prediction_scores: np.ndarray | None,
        buffer_len: int,
    ) -> None:
        """Draw visualization on the image."""
        # Draw all landmarks (white dots)
        for i in range(pixel_landmarks.shape[1]):
            x = int(pixel_landmarks[0][i]) + self.offset_x
            y = int(pixel_landmarks[1][i]) + self.offset_y
            cv2.circle(image, (x, y), 1, Colors.WHITE, 2)

        # Draw jaw line landmarks (red)
        for jaw_idx in self.metrics.get_jaw_indices():
            x = int(pixel_landmarks[0][jaw_idx]) + self.offset_x
            y = int(pixel_landmarks[1][jaw_idx]) + self.offset_y
            cv2.circle(image, (x, y), 2, Colors.RED, 3)

        # Draw cheek width line (blue)
        cheek_left_idx = self.metrics.get_cheek_left_index()
        cheek_right_idx = self.metrics.get_cheek_right_index()
        cheek_left = (
            int(pixel_landmarks[0][cheek_left_idx]) + self.offset_x,
            int(pixel_landmarks[1][cheek_left_idx]) + self.offset_y,
        )
        cheek_right = (
            int(pixel_landmarks[0][cheek_right_idx]) + self.offset_x,
            int(pixel_landmarks[1][cheek_right_idx]) + self.offset_y,
        )
        cv2.line(image, cheek_left, cheek_right, Colors.BLUE, 2)

        # Draw metrics rectangles
        cheek_width = self.metrics.measure_cheek_width(pixel_landmarks)
        jaw_gap = self.metrics.measure_cheek_jaw_gap(pixel_landmarks)

        # Cheek width indicator (blue square)
        cv2.rectangle(
            image,
            (self.canvas_width - 10, self.offset_y + 10),
            (
                self.canvas_width - int(cheek_width * 0.5),
                self.offset_y + 10 + int(cheek_width * 0.5),
            ),
            Colors.BLUE,
            2,
        )
        cv2.putText(
            image,
            f"Cheek width: {cheek_width:.1f}px",
            (self.canvas_width - 220, self.offset_y + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Colors.BLUE,
            2,
        )

        # Jaw gap indicator (red square)
        cv2.rectangle(
            image,
            (self.canvas_width - 10, self.offset_y + 150),
            (
                self.canvas_width - int(jaw_gap * 1.0),
                self.offset_y + 150 + int(jaw_gap * 1.0),
            ),
            Colors.RED,
            2,
        )
        cv2.putText(
            image,
            f"Jaw gap: {jaw_gap:.1f}px",
            (self.canvas_width - 170, self.offset_y + 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Colors.RED,
            2,
        )

        cv2.putText(
            image,
            f"Mouth gap: {mouth_gap:.1f}px",
            (self.canvas_width - 170, self.offset_y + 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Colors.GREEN,
            2,
        )

        # Chewing status
        status_text = ""
        status_color = Colors.WHITE
        # if chewing_flag == 1:
        #     status_text = "FIRST BITE!"
        #     status_color = (0, 255, 255)  # Yellow
        # elif chewing_flag in [2, 3]:
        #     status_text = "CHEWING"
        #     status_color = Colors.GREEN

        # Chew counter
        cv2.putText(
            image,
            f"Chews: {self.chewing_detector.munching_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            Colors.WHITE,
            2,
        )

        cv2.putText(
            image,
            f"Status: {status_text}",
            (10, self.offset_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )

        # ML prediction status/scores
        if self.use_ml_prediction:
            if prediction_scores is None:
                cv2.putText(
                    image,
                    f"ML: buffering {buffer_len}/6",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Colors.YELLOW,
                    2,
                )
            else:
                label_names = ["FIRST_BITE", "ONWARDS", "OTHER"]
                best_idx = int(np.argmax(prediction_scores))
                best_label = label_names[best_idx]
                cv2.putText(
                    image,
                    f"ML: [{prediction_scores[0]:.2f}, {prediction_scores[1]:.2f}, {prediction_scores[2]:.2f}] {best_label}",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Colors.CYAN,
                    2,
                )
        else:
            cv2.putText(
                image,
                "ML: disabled",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                Colors.WHITE,
                2,
            )

        # # Face state
        # face_state_text = ["OK", "No Face", "Face Turned", "Other"][face_state]
        # face_state_color = Colors.GREEN if face_state == 0 else Colors.RED
        # cv2.putText(
        #     image,
        #     f"Face: {face_state_text}",
        #     (10, self.offset_y + 110),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     face_state_color,
        #     2,
        # )

        # Face direction
        cv2.putText(
            image,
            f"Direction: X={face_direction_x:.1f} Y={face_direction_y:.1f}",
            (30, self.offset_y + 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            Colors.WHITE,
            2,
        )

        # Face direction arrows (green) to make yaw/pitch intuitive
        arrow_origin = (140, self.offset_y + 360)
        max_len = 130
        max_angle = 45.0  # Degrees mapped to full arrow length
        dx_norm = max(-1.0, min(1.0, face_direction_x / max_angle))
        dy_norm = max(-1.0, min(1.0, face_direction_y / max_angle))
        arrow_tip = (
            arrow_origin[0] + int(dx_norm * max_len),
            arrow_origin[1] - int(dy_norm * max_len),
        )

        cv2.line(
            image,
            (arrow_origin[0] - max_len, arrow_origin[1]),
            (arrow_origin[0] + max_len, arrow_origin[1]),
            Colors.WHITE,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            image,
            (arrow_origin[0], arrow_origin[1] - max_len),
            (arrow_origin[0], arrow_origin[1] + max_len),
            Colors.WHITE,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(image, arrow_origin, 3, Colors.WHITE, 2)
        cv2.arrowedLine(
            image,
            arrow_origin,
            arrow_tip,
            Colors.GREEN,
            2,
            tipLength=0.3,
        )
        # cv2.putText(
        #     image,
        #     "Yaw (+R / -L)",
        #     (self.canvas_width - 260, self.offset_y + 360),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     Colors.WHITE,
        #     1,
        # )
        # cv2.putText(
        #     image,
        #     "Pitch (+Up / -Dn)",
        #     (self.canvas_width - 260, self.offset_y + 380),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     Colors.WHITE,
        #     1,
        # )


def main():
    """Entry point for chewing detection application."""
    try:
        # Configuration
        use_ml = True  # Set to True to enable ML prediction
        model_dir = "data/tmp_model/exp001"  # Directory containing model pickle files
        model_names = [
            "lgb_0.model",
            "lgb_1.model",
            "lgb_2.model",
            "lgb_3.model",
            "lgb_4.model",
        ]  # List your model files

        app = ChewingDetectionApp(
            camera_index=None,  # Use saved camera index
            use_ml_prediction=use_ml,
            model_dir=model_dir if use_ml else None,
            model_names=model_names if use_ml else None,
        )
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
