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
from util.mediapipe_runner import MediapipeRunner
from util.landmark_metrics import LandmarkMetrics
from util.chewing_detector import ChewingDetector
from util.face_direction import FaceDirectionCalculator
from util.color import Colors


class ChewingDetectionApp:
    """Main application for real-time chewing detection."""

    def __init__(
        self,
        camera_index: int | None = None,
        use_ml_prediction: bool = False,
        model_path: str | None = None,
    ) -> None:
        """Initialize chewing detection application.
        
        Args:
            camera_index: Camera device index (None = use saved config)
            use_ml_prediction: Whether to use ML model for prediction
            model_path: Path to trained ML model file
        """
        # Camera setup
        self.camera_index = camera_index if camera_index is not None else load_camera_index()
        self.v_cap = cv2.VideoCapture(self.camera_index)
        
        if not self.v_cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
        
        # Get frame dimensions
        self.frame_width = int(self.v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.v_cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        
        # Initialize components
        self.metrics = LandmarkMetrics()
        self.chewing_detector = ChewingDetector(
            sequence_length=30,
            firstbite_threshold=-0.5,
            onwards_threshold=-0.3,
        )
        self.face_direction_calc = FaceDirectionCalculator(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )
        
        # ML prediction (optional)
        self.use_ml_prediction = use_ml_prediction
        self.ml_predictor = None
        if use_ml_prediction and model_path:
            # TODO: Implement ML predictor loading
            print(f"Warning: ML prediction requested but not yet implemented")
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
                
                # Convert to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face landmarks
                detection = mp_runner.detect(rgb_image)
                
                if not detection.has_landmarks():
                    cv2.putText(
                        image, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, Colors.RED, 2
                    )
                    cv2.imshow('Chewing Detection', image)
                    
                    if cv2.waitKey(5) & 0xFF == 27:  # ESC
                        break
                    continue
                
                # Get landmarks
                normalized_landmarks = detection.get_normalized_landmarks()
                pixel_landmarks = detection.get_pixel_landmarks()
                
                # Calculate face direction
                face_direction_y, face_direction_x = self.face_direction_calc.calculate_direction(
                    normalized_landmarks
                )
                
                # ML prediction (if enabled)
                prediction_scores = None
                if self.use_ml_prediction and self.ml_predictor is not None:
                    # TODO: Implement prediction
                    pass
                
                # Detect chewing
                chewing_flag, mouth_gap, face_state = self.chewing_detector.detect_chewing(
                    pixel_landmarks,
                    face_direction_x,
                    face_direction_y,
                    prediction_scores,
                )
                
                # Visualize results
                self._draw_visualization(
                    image,
                    pixel_landmarks,
                    chewing_flag,
                    mouth_gap,
                    face_state,
                    face_direction_x,
                    face_direction_y,
                )
                
                # Display
                cv2.imshow('Chewing Detection', image)
                
                self.frame_count += 1
                
                if cv2.waitKey(5) & 0xFF == 27:  # ESC
                    print('Exiting...')
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
    ) -> None:
        """Draw visualization on the image."""
        # Draw all landmarks (white dots)
        for i in range(pixel_landmarks.shape[1]):
            x = int(pixel_landmarks[0][i])
            y = int(pixel_landmarks[1][i])
            cv2.circle(image, (x, y), 1, Colors.WHITE, 2)
        
        # Draw jaw line landmarks (red)
        for jaw_idx in self.metrics.get_jaw_indices():
            x = int(pixel_landmarks[0][jaw_idx])
            y = int(pixel_landmarks[1][jaw_idx])
            cv2.circle(image, (x, y), 2, Colors.RED, 3)
        
        # Draw cheek width line (blue)
        cheek_left_idx = self.metrics.get_cheek_left_index()
        cheek_right_idx = self.metrics.get_cheek_right_index()
        cheek_left = (int(pixel_landmarks[0][cheek_left_idx]), int(pixel_landmarks[1][cheek_left_idx]))
        cheek_right = (int(pixel_landmarks[0][cheek_right_idx]), int(pixel_landmarks[1][cheek_right_idx]))
        cv2.line(image, cheek_left, cheek_right, Colors.BLUE, 2)
        
        # Draw metrics rectangles
        cheek_width = self.metrics.measure_cheek_width(pixel_landmarks)
        jaw_gap = self.metrics.measure_cheek_jaw_gap(pixel_landmarks)
        
        # Cheek width indicator (blue square)
        cv2.rectangle(
            image,
            (self.frame_width - 10, self.frame_height - 10),
            (self.frame_width - int(cheek_width * 0.5), self.frame_height - int(cheek_width * 0.5)),
            Colors.BLUE, 2
        )
        cv2.putText(
            image, "Cheek width", (self.frame_width - 220, self.frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.BLUE, 2
        )
        
        # Jaw gap indicator (red square)
        cv2.rectangle(
            image,
            (self.frame_width - 10, self.frame_height - 150),
            (self.frame_width - int(jaw_gap * 1.0), self.frame_height - 150 - int(jaw_gap * 1.0)),
            Colors.RED, 2
        )
        cv2.putText(
            image, "Jaw gap", (self.frame_width - 180, self.frame_height - 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.RED, 2
        )
        
        # Chewing status
        status_text = "Idle"
        status_color = Colors.WHITE
        if chewing_flag == 1:
            status_text = "FIRST BITE!"
            status_color = (0, 255, 255)  # Yellow
        elif chewing_flag in [2, 3]:
            status_text = "CHEWING"
            status_color = Colors.GREEN
        
        cv2.putText(
            image, f"Status: {status_text}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2
        )
        
        # Chew counter
        cv2.putText(
            image, f"Chews: {self.chewing_detector.munching_count}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.WHITE, 2
        )
        
        # Face state
        face_state_text = ["OK", "No Face", "Face Turned", "Other"][face_state]
        face_state_color = Colors.GREEN if face_state == 0 else Colors.RED
        cv2.putText(
            image, f"Face: {face_state_text}", (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_state_color, 2
        )
        
        # Face direction
        cv2.putText(
            image, f"Direction: X={face_direction_x:.1f} Y={face_direction_y:.1f}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.WHITE, 2
        )
        
        # Mouth gap value
        cv2.putText(
            image, f"Mouth gap: {mouth_gap:.1f}px", (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.WHITE, 2
        )


def main():
    """Entry point for chewing detection application."""
    try:
        app = ChewingDetectionApp(
            camera_index=None,  # Use saved camera index
            use_ml_prediction=False,  # Disable ML for now
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
