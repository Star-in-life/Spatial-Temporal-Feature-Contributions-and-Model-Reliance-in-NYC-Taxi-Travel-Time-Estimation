#!/usr/bin/env python3
"""
run_1m_2015/code/run_factorial_ablation_90k.py
Executes the Complete Multi-Seed Factorial Ablation Matrix on the Stratified 90,000 TRAIN trips:
- 4 Tiers: F0 (Base 7), F1 (Spatial/OSRM 21), F2 (Time/History 20), F3 (Full 34)
- 4 Models: LightGBM, CatBoost, RealMLP, TabM
- 3 Seeds: 42, 43, 44
- Total 48 Refit Experiments on pristine 90k train set (0 overlap with 120k old dataset).
- Evaluated on October DIAG/VAL (10k) and Nov-Dec TEST (20k).
- Outputs:
    results/refits/table_s3_per_seed_ablation.csv (48 rows)
    results/refits/test_ablation_summary.csv (Aggregated Mean ± SD)
"""

import os
import sys
import time
import json
import gc
import psutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Limit thread contention
for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[var] = '4'

try:
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from metrics import compute_metrics_s
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import torch
from pytabkit import RealMLP_TD_Regressor
from pytabkit.models.training.lightning_modules import TabNNModule
from tabm import TabM
import cloudpickle

# Workspaces and directories
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "clean_processed"
SUBSETS_DIR = WORKSPACE / "scaling_subsets"
REFITS_DIR = WORKSPACE / "results/refits"
CK_DIR = REFITS_DIR / "checkpoints"
PRED_DIR = REFITS_DIR / "predictions"
MET_DIR = REFITS_DIR / "metrics"

for d in [REFITS_DIR, CK_DIR, PRED_DIR, MET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)

# Load manifest for exact locked feature definitions
manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
FEATURE_SETS = {
    'F0': manifest['features']['F0'],
    'F1': manifest['features']['F1'],
    'F2': manifest['features']['F2'],
    'F3': manifest['features']['F3'],
}

CATS_GLOBAL = ['VendorID', 'PULocationID', 'DOLocationID', 'pu_borough_enc', 'do_borough_enc']

def get_gpu_temp() -> int:
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True)
        return int(out.strip())
    except Exception:
        return 0

def thermal_guard(threshold=72, target=65, label=""):
    temp = get_gpu_temp()
    if temp >= threshold:
        lbl_str = f" [{label}]" if label else ""
        print(f"[THERMAL GUARD]{lbl_str} GPU reached {temp}C (>= {threshold}C). Pausing for cooldown...")
        while temp > target:
            time.sleep(20)
            temp = get_gpu_temp()
            print(f"  Current GPU Temp: {temp}C (cooldown target: <= {target}C)")
        print(f"[THERMAL GUARD]{lbl_str} Cooldown complete. Resuming execution at {temp}C.")

def minutes_pred(p: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.clip(np.asarray(p).reshape(-1), -20.0, 12.0)))


class TabularPipeline:
    def __init__(self, model_name: str, features: List[str]):
        self.model_name = model_name
        self.features = list(features)
        self.cats = [c for c in CATS_GLOBAL if c in features]
        self.nums = [c for c in features if c not in self.cats]
        self.imp = None
        self.scaler = None
        self.encoder = None
        self.vocab = {}
        self.fallback = {}
        self.mu = 0.0
        self.sd = 1.0
        self.model = None

    def fit_preprocessors(self, df: pd.DataFrame):
        self.imp = SimpleImputer(strategy='median', keep_empty_features=True)
        if self.model_name in ['LightGBM', 'CatBoost']:
            self.imp.fit(df[self.features])
        elif self.model_name == 'RealMLP':
            if len(self.nums) > 0:
                self.scaler = StandardScaler().fit(self.imp.fit_transform(df[self.nums]))
            for c in self.cats:
                vals = df[c].astype(str)
                self.vocab[c] = sorted(vals.unique())
                self.fallback[c] = vals.mode().iloc[0]
        elif self.model_name == 'TabM':
            if len(self.nums) > 0:
                self.scaler = StandardScaler().fit(self.imp.fit_transform(df[self.nums]))
            if len(self.cats) > 0:
                raw_cats = df[self.cats].astype(str)
                self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(raw_cats)
            self.mu = float(df['y'].mean())
            self.sd = max(float(df['y'].std(ddof=0)), 1e-6)

    def transform(self, df: pd.DataFrame):
        if self.model_name in ['LightGBM', 'CatBoost']:
            return self.imp.transform(df[self.features]).astype('float32')
        elif self.model_name == 'RealMLP':
            x = df[self.features].copy()
            for c in self.cats:
                x[c] = x[c].astype(str)
                x[c] = x[c].where(x[c].isin(self.vocab[c]), self.fallback[c])
            if len(self.nums) > 0:
                x[self.nums] = self.scaler.transform(self.imp.transform(x[self.nums]))
            return x
        elif self.model_name == 'TabM':
            if len(self.nums) > 0:
                a = self.scaler.transform(self.imp.transform(df[self.nums])).astype('float32')
            else:
                a = np.zeros((len(df), 0), dtype='float32')
            if len(self.cats) > 0:
                b = self.encoder.transform(df[self.cats].astype(str)).astype('int64') + 1
            else:
                b = np.zeros((len(df), 0), dtype='int64')
            return a, b

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_name in ['LightGBM', 'CatBoost', 'RealMLP']:
            x = self.transform(df)
            raw = self.model.predict(x)
            return minutes_pred(raw)
        elif self.model_name == 'TabM':
            a, b = self.transform(df)
            out = []
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(a), 512):
                    xx = torch.as_tensor(a[start:start+512], device=DEVICE)
                    cc = torch.as_tensor(b[start:start+512], device=DEVICE)
                    out.append(self.model(xx, cc).squeeze(-1).mean(1).cpu().numpy())
            raw = np.concatenate(out) * self.sd + self.mu
            return minutes_pred(raw)

def fit_lightgbm(train_df: pd.DataFrame, pipe: TabularPipeline, seed: int) -> float:
    t0 = time.time()
    pipe.fit_preprocessors(train_df)
    x_tr = pipe.transform(train_df)
    y_tr = train_df['y'].to_numpy(dtype='float32')
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
        random_state=seed
    )
    model = LGBMRegressor(**cfg)
    model.fit(x_tr, y_tr)
    pipe.model = model
    return time.time() - t0

def fit_catboost(train_df: pd.DataFrame, pipe: TabularPipeline, seed: int) -> float:
    t0 = time.time()
    pipe.fit_preprocessors(train_df)
    x_tr = pipe.transform(train_df)
    y_tr = train_df['y'].to_numpy(dtype='float32')
    cfg = dict(
        iterations=1307,
        learning_rate=0.03463288395343894,
        depth=9,
        l2_leaf_reg=0.0,
        random_strength=0.0,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
        random_seed=seed,
        loss_function='RMSE'
    )
    model = CatBoostRegressor(**cfg)
    model.fit(x_tr, y_tr)
    pipe.model = model
    return time.time() - t0

def fit_realmlp(train_df: pd.DataFrame, pipe: TabularPipeline, seed: int) -> float:
    t0 = time.time()
    pipe.fit_preprocessors(train_df)
    x_tr = pipe.transform(train_df)
    y_tr = train_df['y'].to_numpy(dtype='float32')
    fixed_ep = 24

    old_train_end = TabNNModule.on_train_epoch_end
    def train_end_fn(module):
        old_train_end(module)
        if int(module.current_epoch) + 1 >= fixed_ep:
            module.trainer.should_stop = True

    model_kwargs = dict(
        device=DEVICE,
        random_state=seed,
        n_cv=1,
        n_refit=0,
        val_fraction=0.0,
        n_threads=4,
        n_epochs=120,
        batch_size=512,
        hidden_sizes=[256, 256, 256],
        lr=0.14147944317521716,
        wd=0.0018338945994253915,
        p_drop=0.2627636696182051,
        use_early_stopping=False,
        val_metric_name='mae_seconds',
        verbosity=0
    )

    try:
        TabNNModule.on_train_epoch_end = train_end_fn
        model = RealMLP_TD_Regressor(**model_kwargs)
        model.fit(
            x_tr,
            y_tr,
            cat_col_names=pipe.cats,
            val_idxs=np.empty(0, dtype=np.int64)
        )
        pipe.model = model
    finally:
        TabNNModule.on_train_epoch_end = old_train_end
    return time.time() - t0

def fit_tabm(train_df: pd.DataFrame, pipe: TabularPipeline, seed: int) -> float:
    t0 = time.time()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pipe.fit_preprocessors(train_df)
    a, b = pipe.transform(train_df)
    fixed_ep = 39
    cards = [len(v) + 1 for v in pipe.encoder.categories_] if len(pipe.cats) > 0 else []

    xx = torch.as_tensor(a, device=DEVICE)
    cc = torch.as_tensor(b, device=DEVICE)
    yy = torch.as_tensor((train_df['y'].to_numpy(dtype='float32') - pipe.mu) / pipe.sd, device=DEVICE)

    model = TabM.make(
        n_num_features=len(pipe.nums),
        cat_cardinalities=cards,
        d_out=1,
        n_blocks=3,
        d_block=256,
        k=32,
        dropout=0.11061883113135532
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00033070652473618896, weight_decay=0.0)

    for ep in range(1, fixed_ep + 1):
        model.train()
        for ix in torch.randperm(len(train_df), device=DEVICE).split(512):
            pred = model(xx[ix], cc[ix]).squeeze(-1)
            loss = ((pred - yy[ix, None]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pipe.model = model
    return time.time() - t0

TRAINERS = {
    'LightGBM': fit_lightgbm,
    'CatBoost': fit_catboost,
    'RealMLP': fit_realmlp,
    'TabM': fit_tabm
}

def generate_summary_tables():
    """Reads all completed metric JSONs and builds the standard CSV tables without double-rounding."""
    per_seed_rows = []
    raw_rows = []
    
    TIERS = ['F0', 'F1', 'F2', 'F3']
    MODELS = ['LightGBM', 'CatBoost', 'RealMLP', 'TabM']
    SEEDS = [42, 43, 44]

    for tier in TIERS:
        for model in MODELS:
            for seed in SEEDS:
                job = f"{tier}__{model}__{seed}"
                met_file = MET_DIR / f"metric_{job}.json"
                if met_file.exists():
                    d = json.loads(met_file.read_text(encoding="utf-8"))
                    raw_rows.append({
                        'F': tier,
                        'model': model,
                        'seed': seed,
                        'OCT_DIAG_MAE_s': float(d['val']['MAE']),
                        'TEST_MAE_s': float(d['test']['MAE']),
                        'TEST_RMSE_s': float(d['test']['RMSE']),
                        'TEST_MEDAE_s': float(d['test']['MedAE']),
                        'TEST_WAPE': float(d['test']['WAPE']),
                        'TEST_RMSLE': float(d['test']['RMSLE']),
                        'TEST_R2': float(d['test']['R2']),
                        'fit_duration_s': float(d.get('fit_duration_s', 0.0))
                    })
                    per_seed_rows.append({
                        'F': tier,
                        'model': model,
                        'seed': seed,
                        'OCT_DIAG_MAE_s': round(d['val']['MAE'], 2),
                        'TEST_MAE_s': round(d['test']['MAE'], 2),
                        'TEST_RMSE_s': round(d['test']['RMSE'], 2),
                        'TEST_MEDAE_s': round(d['test']['MedAE'], 2),
                        'TEST_WAPE': round(d['test']['WAPE'], 4),
                        'TEST_RMSLE': round(d['test']['RMSLE'], 4),
                        'TEST_R2': round(d['test']['R2'], 4),
                        'fit_duration_s': round(d.get('fit_duration_s', 0.0), 1)
                    })

    if not per_seed_rows:
        return

    df_seed = pd.DataFrame(per_seed_rows)
    df_seed.to_csv(REFITS_DIR / "table_s3_per_seed_ablation.csv", index=False)

    # Aggregated Mean ± SD computed strictly from unrounded float64 metrics
    df_raw = pd.DataFrame(raw_rows)
    agg_rows = []
    metrics_to_agg = ['TEST_MAE_s', 'TEST_RMSE_s', 'TEST_MEDAE_s', 'TEST_WAPE', 'TEST_RMSLE', 'TEST_R2']
    
    for (tier, model), grp in df_raw.groupby(['F', 'model']):
        row = {'F': tier, 'model': model, 'n_seeds': len(grp)}
        for m in metrics_to_agg:
            row[f"{m}_mean"] = float(grp[m].mean())
            row[f"{m}_std"] = float(grp[m].std(ddof=1)) if len(grp) > 1 else 0.0
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    # Sort logically
    tier_order = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3}
    model_order = {'LightGBM': 0, 'CatBoost': 1, 'RealMLP': 2, 'TabM': 3}
    df_agg['t_ord'] = df_agg['F'].map(tier_order)
    df_agg['m_ord'] = df_agg['model'].map(model_order)
    df_agg = df_agg.sort_values(['t_ord', 'm_ord']).drop(columns=['t_ord', 'm_ord'])
    df_agg.to_csv(REFITS_DIR / "test_ablation_summary.csv", index=False)
    print(f"\n[SUMMARY] Saved updated summary tables to {REFITS_DIR}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['LightGBM', 'CatBoost', 'RealMLP', 'TabM'],
                        help='Models to run (default: all 4)')
    parser.add_argument('--tiers', nargs='+', default=['F0', 'F1', 'F2', 'F3'],
                        help='Tiers to run (default: F0, F1, F2, F3)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                        help='Seeds to run (default: 42, 43, 44)')
    parser.add_argument('--force', action='store_true', help='Force re-run existing jobs')
    args = parser.parse_args()

    print("=" * 85)
    print("RUNNING COMPLETE FACTORIAL ABLATION ON STRATIFIED 90k TRAIN DATASET")
    print("=" * 85)
    print(f"Target Models: {args.models}")
    print(f"Target Tiers : {args.tiers}")
    print(f"Target Seeds : {args.seeds}")
    print(f"Device       : {DEVICE}")

    # Load pristine datasets
    t0 = time.time()
    print("\nLoading data...")
    train_df = pd.read_parquet(SUBSETS_DIR / "train_90k.parquet")
    val_df = pd.read_parquet(SUBSETS_DIR / "val_90k.parquet")
    test_df = pd.read_parquet(SUBSETS_DIR / "test_90k.parquet")
    print(f"Data loaded: TRAIN={len(train_df):,}, VAL={len(val_df):,}, TEST={len(test_df):,} in {time.time()-t0:.1f}s.")

    total_jobs = len(args.models) * len(args.tiers) * len(args.seeds)
    job_idx = 0

    for model_name in args.models:
        trainer_fn = TRAINERS[model_name]
        print(f"\n{'#' * 85}")
        print(f"STARTING MODEL: {model_name.upper()}")
        print(f"{'#' * 85}")

        for tier in args.tiers:
            features = FEATURE_SETS[tier]
            for seed in args.seeds:
                job_idx += 1
                job = f"{tier}__{model_name}__{seed}"
                ck_file = CK_DIR / f"refit_{job}.pkl"
                res_file = MET_DIR / f"metric_{job}.json"
                npz_file = PRED_DIR / f"{job}.npz"

                if res_file.exists() and npz_file.exists() and ck_file.exists() and not args.force:
                    saved_m = json.loads(res_file.read_text(encoding="utf-8"))
                    print(f"[{job_idx}/{total_jobs}] Skipping existing {job}: TEST MAE={saved_m['test']['MAE']:.2f}s")
                    continue

                thermal_guard(threshold=72, target=65, label=job)

                print(f"[{job_idx}/{total_jobs}] Fitting {job} (Tier={tier}, Feats={len(features)}, Seed={seed})...")
                pipe = TabularPipeline(model_name, features)

                try:
                    fit_dur = trainer_fn(train_df, pipe, seed)
                    
                    # Inference
                    val_pred = pipe.predict(val_df)
                    test_pred = pipe.predict(test_df)

                    val_m = compute_metrics_s(val_df['dur_min'].to_numpy(), val_pred)
                    test_m = compute_metrics_s(test_df['dur_min'].to_numpy(), test_pred)

                    # Save predictions
                    np.savez_compressed(npz_file,
                                        val_prediction_minutes=val_pred.astype('float32'),
                                        test_prediction_minutes=test_pred.astype('float32'))

                    # Save metrics
                    metric_data = {
                        'job': job,
                        'tier': tier,
                        'model': model_name,
                        'seed': seed,
                        'features_count': len(features),
                        'fit_duration_s': fit_dur,
                        'val': val_m,
                        'test': test_m
                    }
                    res_file.write_text(json.dumps(metric_data, indent=2), encoding="utf-8")

                    # Save checkpoint
                    with open(ck_file, 'wb') as f:
                        cloudpickle.dump({'predictor': pipe, 'job': job}, f)

                    print(f"  -> Finished {job} in {fit_dur:.1f}s | VAL MAE={val_m['MAE']:.2f}s | TEST MAE={test_m['MAE']:.2f}s | R2={test_m['R2']:.4f}")

                    # Generate interim summary table
                    generate_summary_tables()

                except Exception as e:
                    print(f"  [ERROR] Execution failed for {job}: {e}")
                    import traceback
                    traceback.print_exc()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    generate_summary_tables()
    print(f"\n{'=' * 85}")
    print("ALL REQUESTED FACTORIAL ABLATION JOBS COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 85}")

if __name__ == '__main__':
    main()
