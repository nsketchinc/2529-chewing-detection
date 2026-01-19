import platform
import time
from pathlib import Path

import cv2
import numpy as np


def open_camera(index: int) -> cv2.VideoCapture:
    # Windows環境ではCAP_V4L2を使わない
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        # Windowsではデフォルトバックエンドを使用（自動選択）
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed (index={index})")
    return cap


def list_camera_indices(max_fallback: int = 10) -> list[int]:
    indices = []
    if platform.system() == "Linux":
        for path in sorted(Path("/dev").glob("video*")):
            suffix = path.name.replace("video", "")
            if suffix.isdigit():
                indices.append(int(suffix))
    if not indices:
        # Windows等の場合は0から順にチェック
        indices = list(range(max_fallback + 1))
    return indices


def probe_camera(indices: list[int], attempts: int = 5, delay_s: float = 0.1) -> tuple[int, cv2.VideoCapture]:
    for index in indices:
        try:
            cap = open_camera(index)
        except RuntimeError:
            continue
        ok = False
        for _ in range(attempts):
            ok, _frame = cap.read()
            if ok:
                break
            time.sleep(delay_s)
        if ok:
            return index, cap
        cap.release()
    raise RuntimeError(f"No camera produced frames (indices tried: {indices})")


def main() -> None:
    # 利用可能なカメラインデックスを取得
    available_indices = list_camera_indices()
    print(f"Checking available cameras: {available_indices}")
    
    # 最初に使えるカメラを探す
    camera_index = None
    cap = None
    for idx in available_indices:
        try:
            cap = open_camera(idx)
            camera_index = idx
            print(f"Successfully opened camera index {camera_index}")
            break
        except RuntimeError as exc:
            print(f"Camera {idx}: {exc}")
            continue

    window_name = "Camera Test (cv2)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # カメラがない場合はエラーメッセージを表示
    if cap is None or camera_index is None:
        print("No available camera found. Displaying error message.")
        # 黒い画面にエラーメッセージを表示
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.putText(error_frame, "ERROR: No camera detected", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(error_frame, "Please check your camera connection", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(error_frame, "Press 'q' to quit or 0-9 to try camera index", (50, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        while True:
            cv2.imshow(window_name, error_frame)
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord("q"):
                break
            elif ord("0") <= key <= ord("9"):
                # カメラインデックスを試す
                new_index = key - ord("0")
                try:
                    cap = open_camera(new_index)
                    camera_index = new_index
                    print(f"Successfully opened camera index {camera_index}")
                    break
                except RuntimeError as exc:
                    print(f"Camera {new_index}: {exc}")
                    continue
        
        if cap is None:
            cv2.destroyAllWindows()
            return


    print(f"Press q to quit, n/p to switch camera index, 0-9 to select camera directly.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            time.sleep(0.2)
            continue

        # カメラ情報を画面に表示
        info_text = f"Camera Index: {camera_index} (Press n/p or 0-9 to switch)"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            # 次のカメラ
            new_index = camera_index + 1
            cap.release()
            try:
                cap = open_camera(new_index)
                camera_index = new_index
                print(f"Switched to camera index {camera_index}")
            except RuntimeError as exc:
                print(f"Camera {new_index}: {exc}")
                # 元のカメラに戻す
                cap = open_camera(camera_index)
        elif key == ord("p"):
            # 前のカメラ
            if camera_index > 0:
                new_index = camera_index - 1
                cap.release()
                try:
                    cap = open_camera(new_index)
                    camera_index = new_index
                    print(f"Switched to camera index {camera_index}")
                except RuntimeError as exc:
                    print(f"Camera {new_index}: {exc}")
                    # 元のカメラに戻す
                    cap = open_camera(camera_index)
        elif ord("0") <= key <= ord("9"):
            # 数字キーで直接選択
            new_index = key - ord("0")
            if new_index != camera_index:
                cap.release()
                try:
                    cap = open_camera(new_index)
                    camera_index = new_index
                    print(f"Switched to camera index {camera_index}")
                except RuntimeError as exc:
                    print(f"Camera {new_index}: {exc}")
                    # 元のカメラに戻す
                    cap = open_camera(camera_index)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
