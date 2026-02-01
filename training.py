"""Training pipeline for chewing detection models.

Loads raw landmark pickles, builds features matching inference, generates lag
features, trains LightGBM models, and saves pickled models + feature columns.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb


CLASS_FIRST_BITE = 0
CLASS_ONWARDS = 1
CLASS_OTHER = 2


MOUTH_LANDMARKS = [
    0, 10, 11,
    61, 62, 63, 64, 65, 66, 67, 68,
    71, 72, 73, 74, 75, 76, 77, 78,
    164, 165,
]

LABEL_MAP = {
    "1": CLASS_FIRST_BITE,
    1: CLASS_FIRST_BITE,
    "2": CLASS_ONWARDS,
    2: CLASS_ONWARDS,
    "0": CLASS_OTHER,
    0: CLASS_OTHER,
    "-1": CLASS_OTHER,
    -1: CLASS_OTHER,
    "start": CLASS_OTHER,
    "end": CLASS_OTHER,
}


def get_preprocess(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Preprocess raw features.

    Args:
        data: Raw feature array (num_samples, num_features)

    Returns:
        processed_data: Preprocessed feature array
        feat_cols: List of feature column names
    """
    num_samples, num_features = data.shape
    feat_cols = [f"feat_{i}" for i in range(num_features)]
    processed_data = data
    return processed_data, feat_cols


def get_lag_features(
    data: np.ndarray,
    feat_cols: list[str],
    num_lag: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Generate lag features from time series data."""
    lag_features = []
    lag_feat_cols = feat_cols.copy()

    for lag in range(1, num_lag + 1):
        lagged = np.roll(data, lag, axis=0)
        lagged[:lag] = 0
        lag_features.append(lagged)

        for col in feat_cols:
            lag_feat_cols.append(f"{col}_lag{lag}")

    if lag_features:
        lag_data = np.concatenate([data] + lag_features, axis=1)
    else:
        lag_data = data

    return lag_data, lag_feat_cols


def _load_config(path: Path) -> dict[str, Any]:
    print(f"[CONFIG] Loading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[CONFIG] Loaded keys: {list(cfg.keys())}")
    return cfg


def _load_pickle(path: Path) -> dict[str, Any]:
    print(f"[DATA] Loading pickle: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(
        f"[DATA] Loaded keys={list(data.keys())}, landmarks={data['landmarks'].shape}, times={data['now_times'].shape}, labels={data['keys'].shape}"
    )
    return data


def _map_labels(keys: np.ndarray) -> np.ndarray:
    mapped = []
    for k in keys:
        mapped.append(LABEL_MAP.get(k, CLASS_OTHER))
    mapped_arr = np.array(mapped, dtype=np.int32)
    unique, counts = np.unique(mapped_arr, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"[LABEL] Mapped distribution: {dist}")
    return mapped_arr


def _extract_base_features(landmarks: np.ndarray, now_times: np.ndarray) -> np.ndarray:
    """Build base features matching inference (21 landmarks x/y + time delta)."""
    if landmarks.shape[2] > 468:
        landmarks = landmarks[:, :, :468]

    x = landmarks[:, 0, MOUTH_LANDMARKS]
    y = landmarks[:, 1, MOUTH_LANDMARKS]
    times = np.diff(now_times, prepend=1e5)
    data = np.concatenate([x, y, times[:, None]], axis=1)
    return data


def _make_sample_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ones":
        return np.ones_like(labels, dtype=np.float32)

    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    class_weights = {}
    for cls, cnt in zip(unique, counts):
        base = total / (len(unique) * cnt)
        if mode == "equal":
            weight = base
        elif mode == "sqrt":
            weight = np.sqrt(base)
        elif mode == "cbrt":
            weight = np.cbrt(base)
        elif mode == "plus":
            weight = 1.0 + base
        else:
            weight = 1.0
        class_weights[int(cls)] = float(weight)

    weights = np.array([class_weights[int(v)] for v in labels], dtype=np.float32)
    print(f"[WEIGHT] mode={mode}, classes={class_weights}")
    return weights


def _drop_last_munching(labels: np.ndarray, num_drop: int) -> np.ndarray:
    if num_drop <= 0:
        return np.ones_like(labels, dtype=bool)

    mask = np.ones_like(labels, dtype=bool)
    idx = np.where(labels == CLASS_ONWARDS)[0]
    if idx.size > 0:
        drop_idx = idx[-num_drop:]
        mask[drop_idx] = False
    return mask


def build_training_arrays(
    pickle_paths: list[Path],
    lag_num: int,
    drop_last_munching: bool,
    drop_last_munching_num: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_features = []
    all_labels = []
    feat_cols: list[str] = []

    for path in pickle_paths:
        data = _load_pickle(path)
        landmarks = data["landmarks"]  # (N, 3, 468)
        now_times = data["now_times"]
        keys = data["keys"]

        labels = _map_labels(keys)
        if drop_last_munching:
            mask = _drop_last_munching(labels, drop_last_munching_num)
            landmarks = landmarks[mask]
            now_times = now_times[mask]
            labels = labels[mask]
            print(
                f"[DATA] Drop last munching applied: kept {labels.size} samples"
            )

        base = _extract_base_features(landmarks, now_times)
        print(f"[FEATURE] Base features shape: {base.shape}")
        base, base_cols = get_preprocess(base)
        with_lag, feat_cols = get_lag_features(base, base_cols, lag_num)
        print(f"[FEATURE] With lag ({lag_num}) shape: {with_lag.shape}")

        if lag_num > 0 and with_lag.shape[0] > lag_num:
            with_lag = with_lag[lag_num:]
            labels = labels[lag_num:]
            print(f"[FEATURE] Trimmed for lag: {with_lag.shape}, labels={labels.shape}")

        all_features.append(with_lag)
        all_labels.append(labels)

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    print(f"[DATA] Final features shape: {features.shape}")
    print(f"[DATA] Final labels shape: {labels.shape}")
    return features, labels, feat_cols


def train_models() -> None:
    start_time = time.time()
    root = Path(__file__).resolve().parent
    config_path = root / "training_config.yaml"
    config = _load_config(config_path)

    base_cfg = config.get("base_model", {})
    exp_name = config.get("exp_name", "exp001")
    use_pickle_list = base_cfg.get("use_pickle_list", [])
    use_pickle_list_add = base_cfg.get("use_pickle_list_add", [])
    lag_num = int(base_cfg.get("lag_num", 5))
    target_name = base_cfg.get("target_name", "target")
    drop_last_munching = bool(base_cfg.get("drop_last_munching", False))
    drop_last_munching_num = int(base_cfg.get("drop_last_munching_num", 1))
    weight_cfg = base_cfg.get("weight", {})
    tr_weight_mode = weight_cfg.get("tr_weight", "ones")
    va_weight_mode = weight_cfg.get("va_weight", "ones")
    lgb_params = base_cfg.get("lgb_params", {})

    raw_dir = root / "data" / "raw_data"
    pickle_paths = [raw_dir / p for p in (use_pickle_list + use_pickle_list_add)]
    pickle_paths = [p for p in pickle_paths if p.exists()]
    print(f"[DATA] Pickle files: {[p.name for p in pickle_paths]}")
    if not pickle_paths:
        raise FileNotFoundError("No training pickle files found in data/raw_data")

    features, labels, feat_cols = build_training_arrays(
        pickle_paths=pickle_paths,
        lag_num=lag_num,
        drop_last_munching=drop_last_munching,
        drop_last_munching_num=drop_last_munching_num,
    )

    out_dir = root / "data" / "tmp_model" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Saving to: {out_dir}")

    df = pd.DataFrame(features, columns=feat_cols)
    df[target_name] = labels
    df.to_csv(out_dir / "base_training_df.csv", index=False)
    print(f"[OUTPUT] base_training_df.csv saved ({df.shape})")

    with open(out_dir / "feat_cols.pickle", "wb") as f:
        pickle.dump(feat_cols, f)
    print(f"[OUTPUT] feat_cols.pickle saved ({len(feat_cols)} features)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    importances = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(features, labels)):
        x_tr, y_tr = features[tr_idx], labels[tr_idx]
        x_va, y_va = features[va_idx], labels[va_idx]

        tr_weight = _make_sample_weights(y_tr, tr_weight_mode)
        va_weight = _make_sample_weights(y_va, va_weight_mode)

        print(f"[TRAIN] Fold {fold}: train={x_tr.shape}, valid={x_va.shape}")
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            x_tr,
            y_tr,
            sample_weight=tr_weight,
            eval_set=[(x_va, y_va)],
            eval_sample_weight=[va_weight],
        )

        with open(out_dir / f"lgb_{fold}.model", "wb") as f:
            pickle.dump(model, f)
        print(f"[OUTPUT] Model saved: lgb_{fold}.model")

        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

    if importances:
        mean_imp = np.mean(np.vstack(importances), axis=0)
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": mean_imp})
        imp_df.sort_values("importance", ascending=False).to_csv(
            out_dir / "importance.csv", index=False
        )
        print("[OUTPUT] importance.csv saved")

    elapsed = time.time() - start_time
    print(f"Training complete. Models saved to: {out_dir}")
    print(f"[TIME] Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    train_models()
