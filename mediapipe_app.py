import cv2

from util.camera_config import load_camera_index
from util.color import Colors
from util.landmark_metrics import LandmarkMetrics
from util.mediapipe_runner import MediapipeRunner

camera_index = load_camera_index()
v_cap = cv2.VideoCapture(camera_index)  # カメラのIDを選ぶ。映らない場合は番号を変える。

with MediapipeRunner() as mp:
    metrics = LandmarkMetrics()
    while v_cap.isOpened():
        success, image = (
            v_cap.read()
        )  # キャプチャが成功していたら画像データとしてimageに取り込む
        if not success:
            break

        # ビデオモードで処理
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_width, frame_height = image.shape[1], image.shape[0]
        detection = mp.detect(rgb_image)

        if not detection.has_landmarks():
            continue

        # 全ての特徴点を白色で描画
        pixel_landmarks = detection.get_pixel_landmarks()
        for i in range(pixel_landmarks.shape[1]):
            x = int(pixel_landmarks[0][i])
            y = int(pixel_landmarks[1][i])
            cv2.circle(image, center=(x, y), color=Colors.WHITE, radius=1, thickness=2)

        # 特定のポイントを赤点にする
        for jawline in metrics.get_jaw_indices():
            x = int(pixel_landmarks[0][jawline])
            y = int(pixel_landmarks[1][jawline])
            cv2.circle(
                image, center=(x, y), color=Colors.RED, radius=2, thickness=3
            )  # color = BGR

        # 左右の頬の間の距離（青色表示、アゴを開くと"狭く"なる）
        cheek_left_index = metrics.get_cheek_left_index()
        cheek_right_index = metrics.get_cheek_right_index()
        cheek_left = detection.get_pixel_point(cheek_left_index)
        cheek_right = detection.get_pixel_point(cheek_right_index)

        # 左右の頬の間の距離を特徴点上に青色のラインで表示
        cv2.line(
            image,
            pt1=(cheek_left[0], cheek_left[1]),
            pt2=(cheek_right[0], cheek_right[1]),
            color=Colors.BLUE,
            thickness=2,
        )

        # 左右の頬の間の距離の動きを画面左上にブルーの正方形の面積で表示
        cheek_width = metrics.measure_cheek_width(pixel_landmarks)
        cv2.rectangle(
            image,
            pt1=(frame_width - 10, frame_height - 10),
            pt2=(
                frame_width - int(cheek_width * 0.5),
                frame_height - int(cheek_width * 0.5),
            ),
            color=Colors.BLUE,
            thickness=2,
        )  # blue,green,red
        cv2.putText(
            image,
            "Cheek width (blue)",
            (frame_width - 220, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Colors.BLUE,
            2,
        )

        # 左の頬とアゴ先のy座標距離（赤色表示、アゴを開くと"広く"なる）
        jaw_center_index = metrics.get_jaw_tip_index()
        jaw_center = detection.get_pixel_point(jaw_center_index)

        # 左の頬とアゴ先のy座標距離を特徴点上に赤色のラインで表示
        cv2.line(
            image,
            pt1=(jaw_center[0], jaw_center[1]),
            pt2=(jaw_center[0], jaw_center[1]),
            color=Colors.RED,
            thickness=2,
        )  # blue,green,red

        # 左の頬とアゴ先のy座標距離の動きを画面左上にレッドの正方形の面積で表示
        jaw_gap = metrics.measure_cheek_jaw_gap(pixel_landmarks)
        cv2.rectangle(
            image,
            pt1=(frame_width - 10, frame_height - 150),
            pt2=(
                frame_width - int(jaw_gap * 1.0),
                frame_height - 150 - int(jaw_gap * 1.0),
            ),
            color=Colors.RED,
            thickness=2,
        )  # blue,green,red
        cv2.putText(
            image,
            "Cheek-jaw gap (red)",
            (frame_width - 250, frame_height - 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Colors.RED,
            2,
        )

        # distance1,distance2の距離の差を計算
        distance_gap = abs(cheek_width - jaw_gap)

        # distance1,distance2の距離の差を画面左下にグリーンの正方形の面積で表示
        cv2.rectangle(
            image,
            pt1=(frame_width - 10, frame_height - 300),
            pt2=(
                frame_width - int(distance_gap * 0.5),
                frame_height - 300 - int(distance_gap * 0.5),
            ),
            color=Colors.GREEN,
            thickness=2,
        )  # blue,green,red
        cv2.putText(
            image,
            "Gap difference (green)",
            (frame_width - 260, frame_height - 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Colors.GREEN,
            2,
        )

        cv2.imshow("MediaPipe Face Mesh", image)  # フリップせずそのまま表示

        if cv2.waitKey(5) & 0xFF == 27:  # ESCキーが押されたら終わる
            print("終了")
            break

v_cap.release()
cv2.destroyAllWindows()
