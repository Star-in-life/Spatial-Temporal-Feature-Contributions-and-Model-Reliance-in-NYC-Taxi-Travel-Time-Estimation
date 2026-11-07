#!/usr/bin/env python3
"""
run_1m_2015/code/run_all_scaling_stratified.py
Executes the Comprehensive Multi-Seed Scaling Benchmark across all paper models:
OSRM, TEMP, LightGBM, CatBoost, RealMLP, TabM.

Across 7 Stratified Nested Tiers:
S_90k ⊂ S_180k ⊂ S_360k ⊂ S_540k ⊂ S_720k ⊂ S_900k ⊂ S_1080k

Guarantees:
1. Zero Data-Scale Bleeding: Each scale tier S uses its own local train_{S}k, val_{S}k, and test_{S}k.
2. Full Multi-Seed Coverage: 3 deterministic seeds (42, 43, 44) for all machine learning and deep learning models.
3. CatBoost strictly on CPU (thread_count=4) conforming 100% to submission config catboost_best.yaml.
4. Reuses already completed jobs (especially 90k from refits) without repeating computation.
5. Saves individual results and aggregated Mean ± SD summary tables.
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

# Cap threads
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

WORKSPACE = Path("d:/Download/DAP391m/run_1m_2015").resolve()
DATA_DIR = WORKSPACE / "data/2015/processed"
SUBSETS_DIR = WORKSPACE / "data/2015/subsets"
RES_DIR = WORKSPACE / "results/scaling"
CK_DIR = RES_DIR / "checkpoints"
PRED_DIR = RES_DIR / "predictions"
MET_DIR = RES_DIR / "metrics"
REFITS_DIR = WORKSPACE / "results/refits"

for d in [RES_DIR, CK_DIR, PRED_DIR, MET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)

CATS = ['VendorID', 'PULocationID', 'DOLocationID', 'pu_borough_enc', 'do_borough_enc']
TIERS = ['90k', '180k', '360k', '540k', '720k', '900k', '1080k']
SEEDS = [42, 43, 44]

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

def compute_metrics_s(y_true_min: np.ndarray, y_pred_min: np.ndarray) -> Dict[str, float]:
    y_true_s = np.asarray(y_true_min, dtype='float64') * 60.0
    y_pred_s = np.asarray(y_pred_min, dtype='float64') * 60.0
    err = y_true_s - y_pred_s
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    medae = float(np.median(np.abs(err)))
    wape = float(np.sum(np.abs(err)) / np.maximum(np.sum(y_true_s), 1e-8))
    ss_tot = np.sum((y_true_s - np.mean(y_true_s)) ** 2)
    ss_res = np.sum(err ** 2)
    r2 = float(1.0 - ss_res / np.maximum(ss_tot, 1e-8))
    log_true = np.log1p(np.maximum(np.asarray(y_true_min, dtype='float64'), 0.0))
    log_pred = np.log1p(np.maximum(np.asarray(y_pred_min, dtype='float64'), 0.0))
    rmsle = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MedAE': medae,
        'WAPE': wape,
        'RMSLE': rmsle,
        'R2': r2
    }

# ==========================================
# 1. BASELINE RUNNER (OSRM & TEMP)
# ==========================================
class BaselineModel:
    def __init__(self, name: str):
        self.name = name
        self.od = None
        self.slot_median = None
        self.global_median = None
        self.slot_speed = None
        self.global_speed = None
        self.od_normalized = None

    def fit(self, train_df: pd.DataFrame):
        self.slot_median = train_df.groupby('slot')['dur_min'].median()
        self.global_median = float(train_df['dur_min'].median())
        self.od = train_df.groupby(['PULocationID', 'DOLocationID'])['dur_min'].median()

        if self.name == 'TEMP':
            valid_speed_mask = (train_df['haversine_km'] > 0) & (train_df['dur_min'] > 0)
            speed = (train_df['haversine_km'] / (train_df['dur_min'] / 60.0)).where(valid_speed_mask)
            self.slot_speed = speed.groupby(train_df['slot']).mean()
            self.global_speed = float(speed[valid_speed_mask].mean())
            norm = train_df['dur_min'] * train_df['slot'].map(self.slot_speed).fillna(self.global_speed)
            self.od_normalized = train_df.assign(normalized=norm).groupby(['PULocationID', 'DOLocationID'])['normalized'].mean()
        return self

    def baseline_fallback(self, df: pd.DataFrame) -> np.ndarray:
        pair = pd.MultiIndex.from_frame(df[['PULocationID', 'DOLocationID']])
        od_preds = pd.Series(self.od.reindex(pair).to_numpy(), index=df.index)
        slot_preds = df['slot'].map(self.slot_median)
        fallback = od_preds.fillna(slot_preds).fillna(self.global_median).to_numpy()
        return fallback

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.name == 'OSRM':
            ok = df['route_status'].eq('ok').to_numpy() & df['osrm_min'].gt(0).to_numpy()
            return np.where(ok, df['osrm_min'].to_numpy(), self.baseline_fallback(df))
        elif self.name == 'TEMP':
            pair = pd.MultiIndex.from_frame(df[['PULocationID', 'DOLocationID']])
            query_slot_speed = df['slot'].map(self.slot_speed).fillna(self.global_speed).to_numpy()
            query_slot_speed = np.maximum(query_slot_speed, 1e-8)
            p = self.od_normalized.reindex(pair).to_numpy() / query_slot_speed
            return np.where(np.isfinite(p), p, self.baseline_fallback(df))

def run_baselines():
    print("\n" + "=" * 85)
    print("RUNNING BASELINES (OSRM & TEMP) ON ISOLATED SUBSETS")
    print("=" * 85)
    
    for b_name in ['OSRM', 'TEMP']:
        records = []
        for t in TIERS:
            sub_train = pd.read_parquet(SUBSETS_DIR / f"train_{t}.parquet")
            sub_val = pd.read_parquet(SUBSETS_DIR / f"val_{t}.parquet")
            sub_test = pd.read_parquet(SUBSETS_DIR / f"test_{t}.parquet")

            m = BaselineModel(b_name).fit(sub_train)
            v_p = m.predict(sub_val)
            t_p = m.predict(sub_test)
            v_met = compute_metrics_s(sub_val['dur_min'].to_numpy(), v_p)
            t_met = compute_metrics_s(sub_test['dur_min'].to_numpy(), t_p)
            
            rec = {
                'tier': t,
                'train_size': len(sub_train),
                'model': b_name,
                'seed': 42,
                'val_mae_s': v_met['MAE'],
                'val_r2': v_met['R2'],
                'test_mae_s': t_met['MAE'],
                'test_rmse_s': t_met['RMSE'],
                'test_medae_s': t_met['MedAE'],
                'test_wape': t_met['WAPE'],
                'test_rmsle': t_met['RMSLE'],
                'test_r2': t_met['R2']
            }
            records.append(rec)
            print(f"[{b_name:4s}] Tier {t:<6}: TEST MAE={t_met['MAE']:.2f}s | RMSE={t_met['RMSE']:.2f}s | R2={t_met['R2']:.4f}")
        
        b_df = pd.DataFrame(records)
        b_df.to_csv(RES_DIR / f"scaling_{b_name.lower()}_summary.csv", index=False)

# ==========================================
# 2. ML & NEURAL PIPELINE RUNNER
# ==========================================
class TabularPipeline:
    def __init__(self, model_name: str, features: List[str]):
        self.model_name = model_name
        self.features = list(features)
        self.cats = [c for c in CATS if c in features]
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
            self.scaler = StandardScaler().fit(self.imp.fit_transform(df[self.nums]))
            for c in self.cats:
                vals = df[c].astype(str)
                self.vocab[c] = sorted(vals.unique())
                self.fallback[c] = vals.mode().iloc[0]
        elif self.model_name == 'TabM':
            self.scaler = StandardScaler().fit(self.imp.fit_transform(df[self.nums]))
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
            x[self.nums] = self.scaler.transform(self.imp.transform(x[self.nums]))
            return x
        elif self.model_name == 'TabM':
            a = self.scaler.transform(self.imp.transform(df[self.nums])).astype('float32')
            b = self.encoder.transform(df[self.cats].astype(str)).astype('int64') + 1
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

def fit_lightgbm(sub_train: pd.DataFrame, pipe: TabularPipeline, seed: int = 42) -> float:
    t0 = time.time()
    pipe.fit_preprocessors(sub_train)
    x_tr = pipe.transform(sub_train)
    y_tr = sub_train['y'].to_numpy(dtype='float32')
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

def fit_catboost(sub_train: pd.DataFrame, pipe: TabularPipeline, seed: int = 42) -> float:
    t0 = time.time()
    pipe.fit_preprocessors(sub_train)
    x_tr = pipe.transform(sub_train)
    y_tr = sub_train['y'].to_numpy(dtype='float32')
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

def fit_realmlp(sub_train: pd.DataFrame, pipe: TabularPipeline, seed: int = 42) -> float:
    t0 = time.time()
    pipe.fit_preprocessors(sub_train)
    x_tr = pipe.transform(sub_train)
    y_tr = sub_train['y'].to_numpy(dtype='float32')
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

def fit_tabm(sub_train: pd.DataFrame, pipe: TabularPipeline, seed: int = 42) -> float:
    t0 = time.time()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pipe.fit_preprocessors(sub_train)
    a, b = pipe.transform(sub_train)
    fixed_ep = 39
    cards = [len(v) + 1 for v in pipe.encoder.categories_]

    xx = torch.as_tensor(a, device=DEVICE)
    cc = torch.as_tensor(b, device=DEVICE)
    yy = torch.as_tensor((sub_train['y'].to_numpy(dtype='float32') - pipe.mu) / pipe.sd, device=DEVICE)

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
        for ix in torch.randperm(len(sub_train), device=DEVICE).split(512):
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

def update_scaling_summaries():
    """Compiles both per-seed and aggregated Mean ± SD scaling summaries."""
    all_records = []
    
    # 1. Collect ML model records
    for met_file in MET_DIR.glob("*.json"):
        try:
            d = json.loads(met_file.read_text(encoding="utf-8"))
            all_records.append({
                'tier': d['tier'],
                'train_size': d['train_size'],
                'model': d['model'],
                'seed': d.get('seed', 42),
                'test_mae_s': d['test']['MAE'],
                'test_rmse_s': d['test']['RMSE'],
                'test_medae_s': d['test']['MedAE'],
                'test_wape': d['test']['WAPE'],
                'test_rmsle': d['test']['RMSLE'],
                'test_r2': d['test']['R2'],
                'fit_duration_s': d.get('fit_duration_s', 0.0),
                'val_mae_s': d['val']['MAE'],
                'val_r2': d['val']['R2']
            })
        except Exception:
            pass

    # 2. Collect Baselines
    for b_name in ['osrm', 'temp']:
        b_file = RES_DIR / f"scaling_{b_name}_summary.csv"
        if b_file.exists():
            b_df = pd.read_csv(b_file)
            for _, r in b_df.iterrows():
                all_records.append({
                    'tier': r['tier'],
                    'train_size': int(r['train_size']),
                    'model': r['model'],
                    'seed': 42,
                    'test_mae_s': r['test_mae_s'],
                    'test_rmse_s': r['test_rmse_s'],
                    'test_medae_s': r['test_medae_s'],
                    'test_wape': r['test_wape'],
                    'test_rmsle': r['test_rmsle'],
                    'test_r2': r['test_r2'],
                    'fit_duration_s': 0.0,
                    'val_mae_s': r.get('val_mae_s', 0.0),
                    'val_r2': r.get('val_r2', 0.0)
                })

    if not all_records:
        return

    full_df = pd.DataFrame(all_records)
    # Sort order
    tier_map = {t: idx for idx, t in enumerate(TIERS)}
    full_df['t_ord'] = full_df['tier'].map(tier_map)
    full_df = full_df.sort_values(['t_ord', 'model', 'seed']).drop(columns=['t_ord'])
    full_df.to_csv(RES_DIR / "scaling_all_models_summary.csv", index=False)

    # Aggregated Mean ± SD
    agg_rows = []
    for (tier, model), grp in full_df.groupby(['tier', 'model']):
        row = {
            'tier': tier,
            'train_size': grp['train_size'].iloc[0],
            'model': model,
            'n_seeds': len(grp),
            'test_mae_mean': grp['test_mae_s'].mean(),
            'test_mae_std': grp['test_mae_s'].std(ddof=1) if len(grp) > 1 else 0.0,
            'test_rmse_mean': grp['test_rmse_s'].mean(),
            'test_rmse_std': grp['test_rmse_s'].std(ddof=1) if len(grp) > 1 else 0.0,
            'test_r2_mean': grp['test_r2'].mean(),
            'test_r2_std': grp['test_r2'].std(ddof=1) if len(grp) > 1 else 0.0,
            'test_wape_mean': grp['test_wape'].mean(),
            'test_wape_std': grp['test_wape'].std(ddof=1) if len(grp) > 1 else 0.0,
            'fit_duration_mean': grp['fit_duration_s'].mean()
        }
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df['t_ord'] = agg_df['tier'].map(tier_map)
    agg_df = agg_df.sort_values(['t_ord', 'model']).drop(columns=['t_ord'])
    agg_df.to_csv(RES_DIR / "scaling_summary_aggregated.csv", index=False)
    print(f"[SUMMARY] Updated scaling summaries: {len(full_df)} runs, {len(agg_df)} aggregated cells.")

def run_model_scaling(model_name: str, features: List[str]):
    print("\n" + "=" * 90)
    print(f"RUNNING SCALING BENCHMARK FOR MODEL: {model_name.upper()} (90k -> 1,080k, Seeds 42, 43, 44)")
    print("=" * 90)

    trainer_fn = TRAINERS[model_name]

    for tier_label in TIERS:
        sub_train_file = SUBSETS_DIR / f"train_{tier_label}.parquet"
        sub_val_file = SUBSETS_DIR / f"val_{tier_label}.parquet"
        sub_test_file = SUBSETS_DIR / f"test_{tier_label}.parquet"

        for seed in SEEDS:
            job_name = f"{model_name.lower()}_{tier_label}__seed_{seed}"
            ck_path = CK_DIR / f"{job_name}.pkl"
            pred_path = PRED_DIR / f"{job_name}.npz"
            met_path = MET_DIR / f"{job_name}.json"

            # Check if 90k run already exists in results/refits
            if tier_label == '90k':
                refit_met = REFITS_DIR / "metrics" / f"metric_F3__{model_name}__{seed}.json"
                refit_pred = REFITS_DIR / "predictions" / f"F3__{model_name}__{seed}.npz"
                refit_ck = REFITS_DIR / "checkpoints" / f"refit_F3__{model_name}__{seed}.pkl"
                if refit_met.exists() and refit_pred.exists() and refit_ck.exists() and not (met_path.exists() and pred_path.exists() and ck_path.exists()):
                    import shutil
                    d = json.loads(refit_met.read_text(encoding="utf-8"))
                    rec = {
                        'model': model_name,
                        'tier': tier_label,
                        'train_size': 90000,
                        'seed': seed,
                        'fit_duration_s': d.get('fit_duration_s', 0.0),
                        'val': d['val'],
                        'test': d['test']
                    }
                    met_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                    if not pred_path.exists():
                        shutil.copy2(refit_pred, pred_path)
                    if not ck_path.exists():
                        shutil.copy2(refit_ck, ck_path)
                    print(f"  [Reused 90k from refits] {job_name}: TEST MAE={d['test']['MAE']:.2f}s | R2={d['test']['R2']:.4f}")
                    update_scaling_summaries()
                    continue

            if met_path.exists() and pred_path.exists() and ck_path.exists():
                saved_rec = json.loads(met_path.read_text(encoding="utf-8"))
                print(f"  [Found existing] {job_name}: TEST MAE={saved_rec['test']['MAE']:.2f}s")
                continue

            thermal_guard(threshold=72, target=65, label=f"pre-fit {job_name}")

            print(f"\n[{job_name}] Loading {sub_train_file.name} (Seed={seed})...")
            sub_train = pd.read_parquet(sub_train_file)
            sub_val = pd.read_parquet(sub_val_file)
            sub_test = pd.read_parquet(sub_test_file)
            n = len(sub_train)

            pipe = TabularPipeline(model_name, features)

            print(f"  Training {model_name} on {n:,} rows (Seed={seed})...")
            fit_duration = trainer_fn(sub_train, pipe, seed=seed)
            print(f"  Fit finished in {fit_duration:.1f}s ({fit_duration/60.0:.2f} min).")

            # Inference
            val_pred_min = pipe.predict(sub_val)
            val_metrics = compute_metrics_s(sub_val['dur_min'].to_numpy(), val_pred_min)

            test_pred_min = pipe.predict(sub_test)
            test_metrics = compute_metrics_s(sub_test['dur_min'].to_numpy(), test_pred_min)

            print(f"  VAL  (Oct 10k):  MAE = {val_metrics['MAE']:.2f}s | RMSE = {val_metrics['RMSE']:.2f}s | R2 = {val_metrics['R2']:.4f}")
            print(f"  TEST (Nov-Dec):  MAE = {test_metrics['MAE']:.2f}s | RMSE = {test_metrics['RMSE']:.2f}s | R2 = {test_metrics['R2']:.4f}")

            # Save predictions
            np.savez_compressed(
                pred_path,
                val_prediction_minutes=val_pred_min,
                test_prediction_minutes=test_pred_min
            )

            # Save checkpoint
            tmp_ck = ck_path.with_suffix(".tmp")
            with tmp_ck.open("wb") as f:
                cloudpickle.dump({
                    'model_name': model_name,
                    'tier': tier_label,
                    'train_size': n,
                    'seed': seed,
                    'pipeline': pipe,
                    'fit_duration_s': fit_duration
                }, f)
            tmp_ck.replace(ck_path)

            # Save metric JSON
            rec = {
                'model': model_name,
                'tier': tier_label,
                'train_size': n,
                'seed': seed,
                'fit_duration_s': fit_duration,
                'val': val_metrics,
                'test': test_metrics
            }
            met_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

            update_scaling_summaries()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main():
    print("=" * 90)
    print("MASTER MULTI-SEED SCALING BENCHMARK (7 TIERS x 4 MODELS x 3 SEEDS + BASELINES)")
    print("=" * 90)

    # 1. Load Features
    with open(DATA_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    features = manifest["features"]["F3"]
    assert len(features) == 34

    # 2. Run Baselines
    run_baselines()

    # 3. Run Primary Models (LightGBM, CatBoost CPU, RealMLP, TabM)
    for m in ['LightGBM', 'CatBoost', 'RealMLP', 'TabM']:
        run_model_scaling(m, features)

    update_scaling_summaries()
    print("\n" + "=" * 90)
    print("ALL SCALING EXPERIMENTS COMPLETED!")
    print("=" * 90)

if __name__ == "__main__":
    main()
