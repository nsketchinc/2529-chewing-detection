"""Training utilities for feature preprocessing and lag feature generation.

This is a stub/template. Replace with your actual training module if available.
"""
from __future__ import annotations

import numpy as np


def get_preprocess(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Preprocess raw features.
    
    Args:
        data: Raw feature array (num_samples, num_features)
        
    Returns:
        processed_data: Preprocessed feature array
        feat_cols: List of feature column names
    """
    # TODO: Implement your actual preprocessing logic
    # This is a placeholder that returns data as-is
    
    num_samples, num_features = data.shape
    feat_cols = [f"feat_{i}" for i in range(num_features)]
    
    # Example: normalize or standardize features
    # processed_data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    processed_data = data  # Placeholder
    
    return processed_data, feat_cols


def get_lag_features(
    data: np.ndarray,
    feat_cols: list[str],
    num_lag: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Generate lag features from time series data.
    
    Args:
        data: Feature array (num_samples, num_features)
        feat_cols: List of feature column names
        num_lag: Number of lag steps to create
        
    Returns:
        lag_data: Feature array with lag features appended
        lag_feat_cols: Updated list of feature column names
    """
    # TODO: Implement your actual lag feature generation logic
    # This is a placeholder
    
    num_samples, num_features = data.shape
    lag_features = []
    lag_feat_cols = feat_cols.copy()
    
    for lag in range(1, num_lag + 1):
        # Shift data by lag steps
        lagged = np.roll(data, lag, axis=0)
        lagged[:lag] = 0  # Zero-pad the beginning
        lag_features.append(lagged)
        
        # Add column names
        for col in feat_cols:
            lag_feat_cols.append(f"{col}_lag{lag}")
    
    # Concatenate original and lag features
    if lag_features:
        lag_data = np.concatenate([data] + lag_features, axis=1)
    else:
        lag_data = data
    
    return lag_data, lag_feat_cols


def example_usage():
    """Example of how to use these functions."""
    # Create dummy data
    data = np.random.randn(100, 940)  # 100 samples, 940 features (468*2 + 3 + margin)
    
    # Preprocess
    processed, feat_cols = get_preprocess(data)
    print(f"Preprocessed shape: {processed.shape}")
    
    # Add lag features
    with_lags, lag_cols = get_lag_features(processed, feat_cols, num_lag=5)
    print(f"With lags shape: {with_lags.shape}")
    print(f"Number of feature columns: {len(lag_cols)}")


if __name__ == "__main__":
    example_usage()
