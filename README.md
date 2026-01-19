Chewing Detection – Training Guide
=================================

This project supports training LightGBM models for chewing detection using the bundled `training.py` (ported from the original soshaku implementation).

Quick Start (Training)
----------------------
1) Install deps in your venv
	```bash
	pip install numpy pandas lightgbm pyyaml scikit-learn
	```
2) Prepare data
	- Place your training pickles under `data/raw_data/` (filenames listed in `training_config.yaml` → `base_model.use_pickle_list` / `use_pickle_list_add`).
3) Prepare config
	- Put `training_config.yaml` in the project root (same level as `training.py`).
	- Key fields used:
	  - `exp_name`: output folder name under `data/tmp_model/`.
	  - `base_model.use_pickle_list`: training pickle filenames.
	  - `base_model.lag_num`: lag window (default 5, matches inference).
	  - `base_model.target_name`: target column name.
	  - `base_model.lgb_params`: LightGBM params.
4) Run training
	```bash
	python training.py
	```
	Outputs are stored in `data/tmp_model/<exp_name>/`:
	- `lgb_0.model` … `lgb_4.model`: trained models (5-fold)
	- `feat_cols.pickle`: feature ordering used by the models
	- `importance.csv`, `base_training_df.csv`: logs/results

Using Trained Models in main.py
-------------------------------
1) Point `main.py` to your models (example):
	```python
	use_ml = True
	model_dir = "data/tmp_model/<exp_name>"
	model_names = ["lgb_0.model", "lgb_1.model", "lgb_2.model", "lgb_3.model", "lgb_4.model"]
	```
2) Ensure `training.py` is in the project root so inference can reuse the same preprocessing (`get_preprocess`, `get_lag_features`).
3) Run the app
	```bash
	python main.py
	```

Notes
-----
- `face_geometry.py` is leveraged when present for metric landmark conversion (PCF); otherwise a fallback approximation is used.
- During inference, ML scores are averaged across all listed models and passed to `ChewingDetector` for event triggering.
- Early frames (< sequence_length) return no ML scores; geometric checks still run.
