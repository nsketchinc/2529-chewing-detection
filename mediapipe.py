import cv2
from util.camera_config import load_camera_index
from util.face_landmarker_runner import FaceLandmarkerRunner
from util.landmark_metrics import LandmarkMetrics

camera_index = load_camera_index()
v_cap = cv2.VideoCapture(camera_index)#カメラのIDを選ぶ。映らない場合は番号を変える。

with FaceLandmarkerRunner() as landmarker:
    metrics = LandmarkMetrics()
    while v_cap.isOpened():
        success, image = v_cap.read()#キャプチャが成功していたら画像データとしてimageに取り込む
        if not success:
            break
        # オリジナルのカメラ映像を保存しておく
        original_frame = image.copy()

        # ビデオモードで処理
        detection = landmarker.detect(image)
        
        # 元の画像サイズを取得
        frame_height, frame_width = image.shape[:2]
        
        # カメラ映像のコピー上にランドマークを重ね描きする
        image = original_frame.copy()

        # 新しいAPIでは results.face_landmarks がリストになっている
        if detection.task_result.face_landmarks and detection.normalized_landmarks is not None:
            normalized_landmarks = detection.normalized_landmarks
            landmark_metrics = metrics.measure(normalized_landmarks, frame_width, frame_height)
            # 全ての特徴点を白色で描画
            face_points = normalized_landmarks.shape[1]
            for i in range(face_points):
                x = int(normalized_landmarks[0][i] * frame_width)
                y = int(normalized_landmarks[1][i] * frame_height)
                cv2.circle(image, center=(x, y), color=(255, 255, 255), radius=1, thickness=2)# color = BGR
            
            # 特定のポイントを赤点にする
            for jawline in landmarker.metrics.jaw_indices:
                x = int(normalized_landmarks[0][jawline] * frame_width)
                y = int(normalized_landmarks[1][jawline] * frame_height)
                cv2.circle(image, center=(x, y), color=(0, 0, 255), radius=2, thickness=3)# color = BGR

            # 左右の頬の間の距離（青色表示、アゴを開くと"狭く"なる）
            pt1_x, pt1_y = landmark_metrics.cheek_left
            pt2_x, pt2_y = landmark_metrics.cheek_right

            # 左右の頬の間の距離を特徴点上に青色のラインで表示
            cv2.line(image, pt1=(pt1_x, pt1_y), pt2=(pt2_x, pt2_y), color=(255,0,0), thickness=2)#blue,green,red

            # 左右の頬の間の距離の動きを画面左上にブルーの正方形の面積で表示
            cv2.rectangle(image, pt1=(frame_width-10, frame_height-10), 
                         pt2=(frame_width-int(landmark_metrics.cheek_width*0.5), frame_height-int(landmark_metrics.cheek_width*0.5)), 
                         color=(255,0,0), thickness=2)#blue,green,red
            cv2.putText(image, "Cheek width (blue)", (frame_width-220, frame_height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)


            # 左の頬とアゴ先のy座標距離（赤色表示、アゴを開くと"広く"なる）
            jaw_center_x, jaw_center_y = landmark_metrics.jaw_tip
            distance2 = landmark_metrics.cheek_jaw_gap
            
            # 左の頬とアゴ先のy座標距離を特徴点上に赤色のラインで表示
            cv2.line(image, pt1=(jaw_center_x, pt1_y), pt2=(jaw_center_x, jaw_center_y), 
                color=(0,0,255), thickness=2)#blue,green,red

            # 左の頬とアゴ先のy座標距離の動きを画面左上にレッドの正方形の面積で表示
            cv2.rectangle(image, pt1=(frame_width-10, frame_height-150), 
                         pt2=(frame_width-int(distance2*1.0), frame_height-150-int(distance2*1.0)), 
                         color=(0,0,255), thickness=2)#blue,green,red
            cv2.putText(image, "Cheek-jaw gap (red)", (frame_width-250, frame_height-160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # distance1,distance2の距離の差を計算
            distance3 = landmark_metrics.gap_diff
            
            # distance1,distance2の距離の差を画面左下にグリーンの正方形の面積で表示
            cv2.rectangle(image, pt1=(frame_width-10, frame_height-300), 
                         pt2=(frame_width-int(distance3*0.5), frame_height-300-int(distance3*0.5)), 
                         color=(0,255,0), thickness=2)#blue,green,red
            cv2.putText(image, "Gap difference (green)", (frame_width-260, frame_height-310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


            cv2.imshow('MediaPipe Face Mesh', image)  # フリップせずそのまま表示
        
        if cv2.waitKey(5) & 0xFF == 27:#ESCキーが押されたら終わる
            print('終了')
            break

v_cap.release()
cv2.destroyAllWindows()

