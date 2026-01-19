#!/usr/bin/env python
"""Inspect the LightGBM model to see expected features."""
import pickle
import sys

model_path = "data/model/009/lgb_1.model"

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model type: {type(model)}")
    print(f"Number of features expected: {model.num_feature()}")
    
    # Try to get more info
    if hasattr(model, 'feature_names'):
        print(f"Feature names (first 10): {model.feature_names()[:10]}")
        print(f"Total feature names: {len(model.feature_names())}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
