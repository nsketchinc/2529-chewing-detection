"""カメラ設定の保存と読み込み機能を提供するモジュール"""
from pathlib import Path


def get_config_path() -> Path:
    """設定ファイルのパスを取得"""
    return Path(__file__).parent.parent / "camera_config.txt"


def save_camera_index(index: int) -> None:
    """カメラインデックスをファイルに保存
    
    Args:
        index: カメラインデックス番号
    """
    config_path = get_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(str(index))
    print(f"カメラインデックス {index} を保存しました")


def load_camera_index(default: int = 1) -> int:
    """保存されたカメラインデックスを読み取る
    
    Args:
        default: ファイルが存在しない場合のデフォルト値
        
    Returns:
        カメラインデックス番号
    """
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                index = int(f.read().strip())
                print(f"カメラインデックス {index} を設定ファイルから読み込みました")
                return index
        except (ValueError, IOError) as e:
            print(f"設定ファイルの読み込みに失敗: {e}")
    print(f"デフォルトのカメラインデックス {default} を使用します")
    return default
