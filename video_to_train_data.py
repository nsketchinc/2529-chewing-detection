import sys
from pathlib import Path
import pickle
from collections import Counter
import cv2
import numpy as np
import srt
from tqdm import tqdm

from util.face_direction import FaceDirectionCalculator
from util.mediapipe_runner import MediapipeRunner
from util.ml_predictor import MetricLandmarkConverter


CONTENT_DICT = {
    "<b>Bite_Start</b>": "1",
    "<b>Munching</b>": "2",
    "<b>START</b>": "start",
    "<b>END</b>": "end",
    0: "0",
    "<b>x</b>": "-1",
}


class MEDIAPIPE():

    def __init__(self, save_dir=None):
        print("[INIT] Opening video...")
        v_cap = cv2.VideoCapture(VIDEO_DEVICE_ID)  # カメラのIDを選ぶ。映らない場合は番号を変える。

        success, image = v_cap.read()
        if not success:
            print("[WARN] Failed to read first frame.")
        self.FRAME_HEIGHT = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 実際のHEIGHT値を読み込む
        self.FRAME_WIDTH = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 実際のWIDTH値を読み込む
        self.FPS = v_cap.get(cv2.CAP_PROP_FPS)
        if self.FPS <= 0:
            self.FPS = 30.0
            print("[WARN] FPS not detected. Fallback to 30 FPS.")
        print(f'VIDEO_DEVICE_ID -> {VIDEO_DEVICE_ID}')
        print(f'(height, width) -> ({self.FRAME_HEIGHT}, {self.FRAME_WIDTH})')
        print(f'FPS             -> {self.FPS}')

        self.FACE_POINTS = 468  # 顔の特徴点の数
        self.DCOFF = 30  # ポイントの描画時の倍数
        v_cap.release()
        self.data = []
        self.times = 0
        self.save_dir = Path('__file__').parent.parent / 'data'
        self.face_direction_calc = FaceDirectionCalculator(
            frame_width=self.FRAME_WIDTH,
            frame_height=self.FRAME_HEIGHT,
        )
        self.metric_converter = MetricLandmarkConverter(
            frame_width=self.FRAME_WIDTH,
            frame_height=self.FRAME_HEIGHT,
            focal_length_y=1750.0,
        )
        print("[INIT] Components initialized.")

    def get_zimaku_data(self, sub_titles):
        """字幕機能で作成したデータを、リスト形式に変換する
        フレーム数は、ビデオのフレーム数から取得
        アノテーションは、(start, end, label)が要素のリストに入れたのち、
        フレーム数と１対１対応するリストに入れる

        Args:
            sub_tiles: 字幕データ

        Returns:
            int: フレーム数
            list: アノテーションのリスト
        """
        print("[ZIMAKU] Loading subtitle data...")
        v_cap = cv2.VideoCapture(VIDEO_DEVICE_ID)
        fps = v_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = self.FPS
            print("[ZIMAKU] FPS not detected for subtitle mapping. Using fallback.")

        # 総フレーム数
        num_all_frame = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[ZIMAKU] Total frames: {num_all_frame}, FPS: {fps}")

        annotation = []
        now = 0
        for sub in sub_titles:
            start = sub.start.seconds + sub.start.microseconds / 1000000
            end = sub.end.seconds + sub.end.microseconds / 1000000
            label = sub.content
            annotation.append((now, start, 0))
            annotation.append((start, end, label))
            now = end

        # Fill tail segment to the end of the video to avoid index overflow
        end_time = num_all_frame / fps
        if now < end_time:
            annotation.append((now, end_time, 0))

        now = 0
        labels = []
        print("[ZIMAKU] Mapping labels to frames...")
        for num_frame in range(num_all_frame):
            time_ = num_frame / fps

            while True:
                if now >= len(annotation):
                    labels.append(0)
                    break
                if annotation[now][0] <= time_ < annotation[now][1]:
                    labels.append(annotation[now][2])
                    break
                else:
                    now += 1

        print("[ZIMAKU] Label mapping complete.")
        return num_all_frame, labels

    def make_train_data(self, num_all_frame, labels):

        v_cap = cv2.VideoCapture(VIDEO_DEVICE_ID)

        frame_interval_ms = int(1000 / self.FPS) if self.FPS > 0 else 33
        print(f"[TRAIN] Starting processing: {num_all_frame} frames")
        print(f"[TRAIN] Frame interval (ms): {frame_interval_ms}")

        with MediapipeRunner(
            model_path='data/face_landmarker.task',
            max_faces=1,
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            frame_interval_ms=frame_interval_ms,
        ) as mp_runner:

            skipped_no_face = 0
            for frame in tqdm(range(num_all_frame)):

                label = labels[frame]
                key_return = CONTENT_DICT[label]
                success, image = v_cap.read()  # キャプチャが成功していたら画像データとしてimageに取り込む
                if not success:
                    print(f"[WARN] Frame read failed at {frame}. Stopping.")
                    break

                now_time = frame / self.FPS
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB形式に変換
                detection = mp_runner.detect(rgb_image)

                if not detection.has_landmarks():
                    skipped_no_face += 1
                    if skipped_no_face % 100 == 0:
                        print(f"[TRAIN] No face detected: {skipped_no_face} frames skipped")
                    continue

                normalized_landmarks = detection.get_normalized_landmarks()  # (3, N)
                face_direction_y, face_direction_x = self.face_direction_calc.calculate_direction(
                    normalized_landmarks
                )

                metric_landmarks = self.metric_converter.convert(normalized_landmarks)

                self._append_data(metric_landmarks, now_time, key_return, face_direction_x, face_direction_y)

            self._save_data(surfix=None)
            print('save, done')
            print(f"[TRAIN] Total skipped frames (no face): {skipped_no_face}")

        v_cap.release()

    def _append_data(self, metric_landmarks, now_time, key, face_direction_x, face_direction_y):
        self.data.append((metric_landmarks, now_time, key, face_direction_x, face_direction_y))

    def _save_data(self, surfix=None):
        landmarks = []
        now_times = []
        keys = []
        x_list = []
        y_list = []
        for landmark, now_time, key, x, y in self.data:
            landmarks.append(landmark)
            now_times.append(now_time)
            keys.append(key)
            x_list.append(x)
            y_list.append(y)
        print(Counter(keys))
        save_data = {
            'landmarks': np.stack(landmarks),
            'now_times': np.array(now_times),
            'keys': np.array(keys),
            'xx': np.array(x_list),
            'yy': np.array(y_list)
        }

        self._dump_data(save_data, VIDEO_ID, surfix)

    def _dump_data(self, data, save_path, surfix):
        (self.save_dir / 'raw_data').mkdir(exist_ok=True)
        if surfix is None:
            file_name = f'raw_data/{save_path}.pickle'
        else:
            file_name = f'raw_data/{save_path}_{surfix}.pickle'
        with open(self.save_dir / file_name, 'wb') as f:
            pickle.dump(data, f)


def check_sub_title(sub_titles):
    # 全部 CONTENT_DICTのものか
    for i, sub in enumerate(sub_titles):
        print(i, sub)
        assert sub.content in CONTENT_DICT.keys(), print(sub.content, i + 1)

    # はじめがBITE STARTか
    assert sub_titles[0].content == "<b>START</b>"

    # 最後が"<b>END</b>"か
    assert sub_titles[-1].content == "<b>END</b>"



if __name__ == "__main__":
    argv = sys.argv
    VIDEO_ID = argv[1]
    print(f'VIDEO_ID: {VIDEO_ID}')

    VIDEO_DEVICE_ID = f"video_data/{VIDEO_ID}.mp4"
    VIDEO_SUB_TITLES_PATH = f"video_data/{VIDEO_ID}.srt"

    print(f"[MAIN] Video path: {VIDEO_DEVICE_ID}")
    print(f"[MAIN] Subtitle path: {VIDEO_SUB_TITLES_PATH}")
    print("[MAIN] Loading subtitle file...")

    with open(VIDEO_SUB_TITLES_PATH, mode='r', encoding="utf-8") as f:
        print("[MAIN] Parsing subtitle file...")
        sub_titles = list(srt.parse(f.read()))

    print(f"[MAIN] Parsed subtitles: {len(sub_titles)} entries")

    print("[MAIN] Validating subtitles...")
    check_sub_title(sub_titles)
    print("[MAIN] Subtitle validation OK")

    print("[MAIN] Initializing processing...")
    media = MEDIAPIPE()
    num_all_frame, labels = media.get_zimaku_data(sub_titles)
    print("[MAIN] Generating training data...")
    media.make_train_data(num_all_frame, labels)
