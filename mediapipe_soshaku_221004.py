import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from util.camera_config import load_camera_index

camera_index = load_camera_index()
v_cap = cv2.VideoCapture(camera_index)#カメラのIDを選ぶ。映らない場合は番号を変える。

# FaceLandmarkerの設定
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # VIDEO モードを使用
    num_faces=1,  # 顔検出の最大数
    min_face_detection_confidence=0.5,  # 顔検出の最小信頼値
    min_face_presence_confidence=0.5,  # 顔存在の最小信頼値
    min_tracking_confidence=0.5  # トラッキングの最小信頼値
)

with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
    frame_timestamp_ms = 0
    
    while v_cap.isOpened():
        success, image = v_cap.read()#キャプチャが成功していたら画像データとしてimageに取り込む
        if not success:
            break
        # オリジナルのカメラ映像を保存しておく
        original_frame = image.copy()
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB形式に変換
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # ビデオモードで処理（タイムスタンプを渡す）
        results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33  # 約30fpsを想定（1000ms/30fps ≈ 33ms）
        
        # 元の画像サイズを取得
        frame_height, frame_width = image.shape[:2]
        
        # カメラ映像のコピー上にランドマークを重ね描きする
        image = original_frame.copy()

        # 新しいAPIでは results.face_landmarks がリストになっている
        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]  # 最初の顔のランドマークを取得
                
            #print(face_landmarks)

            landmarks = [[],[],[]]

            for landmark in face_landmarks:
                landmarks[0].append(landmark.x)
                landmarks[1].append(landmark.y)
                landmarks[2].append(landmark.z)
                
            landmarks = np.array(landmarks)
            
            face_points = 468 #顔の特徴点の数
            
            # 全ての特徴点を白色で描画
            for i in range(face_points):
                x = int(landmarks[0][i] * frame_width)
                y = int(landmarks[1][i] * frame_height)
                cv2.circle(image, center=(x, y), color=(255, 255, 255), radius=1, thickness=2)# color = BGR
            
            # 特定のポイントを赤点にする
            Jawline1=np.array([187, 214, 210, 211, 32, 208, 199, 428, 262, 431, 430, 434, 411])#下アゴのラインを仮に
            
            for jawline in Jawline1:
                x = int(landmarks[0][jawline] * frame_width)
                y = int(landmarks[1][jawline] * frame_height)
                cv2.circle(image, center=(x, y), color=(0, 0, 255), radius=2, thickness=3)# color = BGR
            
            # 左右の頬の間の距離（青色表示、アゴを開くと"狭く"なる）
            pt1_x = int(landmarks[0][Jawline1[0]] * frame_width)
            pt1_y = int(landmarks[1][Jawline1[0]] * frame_height)
            pt2_x = int(landmarks[0][Jawline1[12]] * frame_width)
            pt2_y = int(landmarks[1][Jawline1[12]] * frame_height)
            
            distance1 = np.linalg.norm(np.array([pt1_x, pt1_y]) - np.array([pt2_x, pt2_y]))

            # 左右の頬の間の距離を特徴点上に青色のラインで表示
            cv2.line(image, pt1=(pt1_x, pt1_y), pt2=(pt2_x, pt2_y), color=(255,0,0), thickness=2)#blue,green,red

            # 左右の頬の間の距離の動きを画面左上にブルーの正方形の面積で表示
            cv2.rectangle(image, pt1=(frame_width-10, frame_height-10), 
                         pt2=(frame_width-int(distance1*0.5), frame_height-int(distance1*0.5)), 
                         color=(255,0,0), thickness=2)#blue,green,red
            cv2.putText(image, "Cheek width (blue)", (frame_width-220, frame_height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)


            # 左の頬とアゴ先のy座標距離（赤色表示、アゴを開くと"広く"なる）
            jaw_center_y = int(landmarks[1][Jawline1[6]] * frame_height)
            distance2 = abs(pt1_y - jaw_center_y)
            
            # 左の頬とアゴ先のy座標距離を特徴点上に赤色のラインで表示
            jaw_center_x = int(landmarks[0][Jawline1[6]] * frame_width)
            cv2.line(image, pt1=(jaw_center_x, pt1_y), pt2=(jaw_center_x, jaw_center_y), 
                    color=(0,0,255), thickness=2)#blue,green,red

            # 左の頬とアゴ先のy座標距離の動きを画面左上にレッドの正方形の面積で表示
            cv2.rectangle(image, pt1=(frame_width-10, frame_height-150), 
                         pt2=(frame_width-int(distance2*1.0), frame_height-150-int(distance2*1.0)), 
                         color=(0,0,255), thickness=2)#blue,green,red
            cv2.putText(image, "Cheek-jaw gap (red)", (frame_width-250, frame_height-160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # distance1,distance2の距離の差を計算
            distance3 = abs(distance1 - distance2)
            
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

