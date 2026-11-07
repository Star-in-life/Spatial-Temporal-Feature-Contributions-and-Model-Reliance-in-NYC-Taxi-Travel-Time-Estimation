#!/usr/bin/env python3
"""
run_1m_2015/code/run_table_s5_omit_month.py
Post-hoc check omitting pickup_month from F2 and F3 (Paper Table S5):
Evaluates LightGBM at seed=42 on train_90k.parquet and evaluates on:
1. OCT DIAG (10k October 2015)
2. TEST (20k Nov-Dec 2015)

Outputs:
  run_1m_2015/results/diagnostics/table_s5_no_month_check.csv
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "clean_processed"
SUBSETS_DIR = WORKSPACE / "scaling_subsets"
DIAG_DIR = WORKSPACE / "results/diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

f2_cols = manifest["features"]["F2"]
f2_no_month = [c for c in f2_cols if c != "month"]
assert len(f2_no_month) == 19

f3_cols = manifest["features"]["F3"]
f3_no_month = [c for c in f3_cols if c != "month"]
assert len(f3_no_month) == 33

print("Loading 90k subsets...")
train_df = pd.read_parquet(SUBSETS_DIR / "train_90k.parquet")
val_df = pd.read_parquet(SUBSETS_DIR / "val_90k.parquet")
test_df = pd.read_parquet(SUBSETS_DIR / "test_90k.parquet")

def compute_metrics(y_true_min, y_pred_min):
    y_true_s = np.asarray(y_true_min, dtype=np.float64) * 60.0
    y_pred_s = np.asarray(y_pred_min, dtype=np.float64) * 60.0
    err = y_true_s - y_pred_s
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    medae = float(np.median(np.abs(err)))
    wape = float(np.sum(np.abs(err)) / np.maximum(np.sum(y_true_s), 1e-8))
    ss_tot = np.sum((y_true_s - np.mean(y_true_s)) ** 2)
    ss_res = np.sum(err ** 2)
    r2 = float(1.0 - ss_res / np.maximum(ss_tot, 1e-8))
    log_true = np.log1p(np.maximum(y_true_min, 0.0))
    log_pred = np.log1p(np.maximum(y_pred_min, 0.0))
    rmsle = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MedAE': medae,
        'WAPE': wape,
        'RMSLE': rmsle,
        'R2': r2
    }

cfg = dict(
    boosting_type='gbdt',
    n_estimators=2500,
    learning_rate=0.01238935393797345,
    max_depth=9,
    num_leaves=96,
    min_child_samples=10,
    min_child_weight=0.001,
    min_split_gain=0.0,
    reg_alpha=4.333674922378128,
    reg_lambda=0.0,
    subsample=0.85,
    subsample_freq=0,
    colsample_bytree=0.85,
    n_jobs=4,
    verbosity=-1,
    random_state=42
)

y_tr = train_df['y'].to_numpy(dtype='float32')

# 1. Fit LightGBM on F2 without month
imp_f2 = SimpleImputer(strategy='median')
x_tr_f2 = imp_f2.fit_transform(train_df[f2_no_month]).astype('float32')
x_val_f2 = imp_f2.transform(val_df[f2_no_month]).astype('float32')
x_test_f2 = imp_f2.transform(test_df[f2_no_month]).astype('float32')

print("Fitting LightGBM (seed=42) on F2 without 'month'...")
t0 = time.time()
m_f2 = LGBMRegressor(**cfg)
m_f2.fit(x_tr_f2, y_tr)
fit_time_f2 = time.time() - t0

val_pred_f2 = np.maximum(0.0, np.expm1(np.clip(m_f2.predict(x_val_f2), -20.0, 12.0)))
test_pred_f2 = np.maximum(0.0, np.expm1(np.clip(m_f2.predict(x_test_f2), -20.0, 12.0)))
val_m_f2 = compute_metrics(val_df['dur_min'].to_numpy(), val_pred_f2)
test_m_f2 = compute_metrics(test_df['dur_min'].to_numpy(), test_pred_f2)

# 2. Fit LightGBM on F3 without month
imp_f3 = SimpleImputer(strategy='median')
x_tr_f3 = imp_f3.fit_transform(train_df[f3_no_month]).astype('float32')
x_val_f3 = imp_f3.transform(val_df[f3_no_month]).astype('float32')
x_test_f3 = imp_f3.transform(test_df[f3_no_month]).astype('float32')

print("Fitting LightGBM (seed=42) on F3 without 'month'...")
t0 = time.time()
m_f3 = LGBMRegressor(**cfg)
m_f3.fit(x_tr_f3, y_tr)
fit_time_f3 = time.time() - t0

val_pred_f3 = np.maximum(0.0, np.expm1(np.clip(m_f3.predict(x_val_f3), -20.0, 12.0)))
test_pred_f3 = np.maximum(0.0, np.expm1(np.clip(m_f3.predict(x_test_f3), -20.0, 12.0)))
val_m_f3 = compute_metrics(val_df['dur_min'].to_numpy(), val_pred_f3)
test_m_f3 = compute_metrics(test_df['dur_min'].to_numpy(), test_pred_f3)

# Read unrounded baseline F1, F2 (with month) for seed 42 from metric JSONs
refit_met_dir = WORKSPACE / "results/refits/metrics"
f1_unrounded = float(json.loads((refit_met_dir / "metric_F1__LightGBM__42.json").read_text(encoding="utf-8"))["test"]["MAE"])
f2_unrounded = float(json.loads((refit_met_dir / "metric_F2__LightGBM__42.json").read_text(encoding="utf-8"))["test"]["MAE"])

records = [
    {
        'Feature_Set': 'F1 (full, 21 features)',
        'Model': 'LightGBM',
        'Seed': 42,
        'Omitted_Feature': '',
        'TEST_MAE_s': round(f1_unrounded, 2),
        'TEST_MAE_unrounded_s': f1_unrounded
    },
    {
        'Feature_Set': 'F2 (full, 20 features)',
        'Model': 'LightGBM',
        'Seed': 42,
        'Omitted_Feature': '',
        'TEST_MAE_s': round(f2_unrounded, 2),
        'TEST_MAE_unrounded_s': f2_unrounded
    },
    {
        'Feature_Set': 'F2 without month (19 features)',
        'Model': 'LightGBM',
        'Seed': 42,
        'Omitted_Feature': 'month',
        'TEST_MAE_s': round(test_m_f2['MAE'], 2),
        'TEST_MAE_unrounded_s': test_m_f2['MAE']
    },
    {
        'Feature_Set': 'F3 without month (33 features)',
        'Model': 'LightGBM',
        'Seed': 42,
        'Omitted_Feature': 'month',
        'TEST_MAE_s': round(test_m_f3['MAE'], 2),
        'TEST_MAE_unrounded_s': test_m_f3['MAE']
    }
]

res_df = pd.DataFrame(records)
out_csv = DIAG_DIR / "table_s5_no_month_check.csv"
res_df.to_csv(out_csv, index=False)
print("\n=== TABLE S5 COMPLETE ===")
print(res_df.to_string(index=False))
