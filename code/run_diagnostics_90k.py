#!/usr/bin/env python3
"""
run_1m_2015/code/run_diagnostics_90k.py
Executes Phase 5 Statistical Diagnostics strictly on the Stratified 90,000 TRAIN models:
1. Direct Contrast (Table V): F1 vs F2 61-day Pickup-Day Block Bootstrap (5,000 resamples, seed=20261026).
2. Paired Comparisons on F3 TEST (Table S1): RealMLP vs OSRM, TEMP, LightGBM, CatBoost, TabM.
3. Grouped Permutation on October DIAG/VAL (Table VII): 5 feature groups, 10 permutations (seed=20261023), 5,000 day resamples (seed=20261024).

Outputs:
  run_1m_2015/results/diagnostics/temporal_vs_spatial_direct_contrast.csv
  run_1m_2015/results/diagnostics/paired_comparisons.csv
  run_1m_2015/results/diagnostics/grouped_permutation.csv
"""

import sys
import json
import time
import cloudpickle
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "clean_processed"
SUBSETS_DIR = WORKSPACE / "scaling_subsets"
REFITS_DIR = WORKSPACE / "results/refits"
PRED_DIR = REFITS_DIR / "predictions"
CK_DIR = REFITS_DIR / "checkpoints"
DIAG_DIR = WORKSPACE / "results/diagnostics"

MODELS = ['LightGBM', 'CatBoost', 'RealMLP', 'TabM']
SEEDS = [42, 43, 44]

# Baseline model for OSRM & TEMP on 90k
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

def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 85)
    print("PHASE 5: STATISTICAL CONTRASTS & DIAGNOSTICS ON 90k TRAIN DATASET")
    print("=" * 85)

    test_df = pd.read_parquet(SUBSETS_DIR / "test_90k.parquet")
    val_df = pd.read_parquet(SUBSETS_DIR / "val_90k.parquet")
    train_90k_df = pd.read_parquet(SUBSETS_DIR / "train_90k.parquet")

    truth_test_min = test_df['dur_min'].to_numpy(dtype=np.float64)
    truth_val_min = val_df['dur_min'].to_numpy(dtype=np.float64)

    # 61-day Pickup-Day Block Bootstrap Setup
    pickup_day_test = pd.factorize(test_df['pickup_datetime'].dt.normalize(), sort=True)[0]
    n_days_test = int(pickup_day_test.max()) + 1
    assert n_days_test == 61, f"Expected 61 test days, got {n_days_test}"

    counts_test = np.bincount(pickup_day_test, minlength=n_days_test)
    draws_test = np.random.default_rng(20261026).integers(0, n_days_test, size=(5000, n_days_test))
    den_test = counts_test[draws_test].sum(axis=1)

    def load_test_losses_s(tier: str, model_name: str) -> np.ndarray:
        per_seed = []
        for s in SEEDS:
            npz_path = PRED_DIR / f"{tier}__{model_name}__{s}.npz"
            with np.load(npz_path) as d:
                pred = np.asarray(d['test_prediction_minutes'], dtype=np.float64)
            per_seed.append(np.abs(pred - truth_test_min) * 60.0)
        return np.mean(np.stack(per_seed, axis=0), axis=0)

    # -------------------------------------------------------------
    # 1. DIRECT CONTRAST (TABLE V)
    # -------------------------------------------------------------
    print("\n--- 1. DIRECT CONTRAST: TEMPORAL (F2) VS SPATIAL (F1) ON TEST ---")
    direct_rows = []
    for m in MODELS:
        f0_l = load_test_losses_s('F0', m)
        f1_l = load_test_losses_s('F1', m)
        f2_l = load_test_losses_s('F2', m)

        spatial_gain = float(np.mean(f0_l - f1_l))
        temporal_gain = float(np.mean(f0_l - f2_l))
        diff = f1_l - f2_l
        point = float(np.mean(diff))

        daily_sum = np.bincount(pickup_day_test, weights=diff, minlength=n_days_test)
        boot = daily_sum[draws_test].sum(axis=1) / den_test

        direct_rows.append({
            'model': m,
            'spatial_gain_s': spatial_gain,
            'temporal_gain_s': temporal_gain,
            'temporal_minus_spatial_gain_s': point,
            'point_low_s': float(np.quantile(boot, 0.025)),
            'point_high_s': float(np.quantile(boot, 0.975)),
            'family_low_s': float(np.quantile(boot, 0.025 / 4.0)),
            'family_high_s': float(np.quantile(boot, 1.0 - 0.025 / 4.0))
        })

    direct_df = pd.DataFrame(direct_rows)
    direct_df.to_csv(DIAG_DIR / "temporal_vs_spatial_direct_contrast.csv", index=False)
    print(direct_df.to_string(index=False))

    # -------------------------------------------------------------
    # 2. PAIRED COMPARISONS ON F3 TEST (TABLE S1)
    # -------------------------------------------------------------
    print("\n--- 2. PAIRED COMPARISONS ON F3 TEST (CANDIDATE: RealMLP) ---")
    f3_losses = {m: load_test_losses_s('F3', m) for m in MODELS}

    # Baselines on 90k
    for b_name in ['OSRM', 'TEMP']:
        m_base = BaselineModel(b_name).fit(train_90k_df)
        b_pred = m_base.predict(test_df)
        f3_losses[b_name] = np.abs(np.asarray(b_pred, dtype=np.float64) - truth_test_min) * 60.0

    paired_rows = []
    for ref_m in ['OSRM', 'TEMP', 'LightGBM', 'CatBoost', 'TabM']:
        diff = f3_losses['RealMLP'] - f3_losses[ref_m]
        daily_sum = np.bincount(pickup_day_test, weights=diff, minlength=n_days_test)
        boot = daily_sum[draws_test].sum(axis=1) / den_test

        paired_rows.append({
            'candidate': 'RealMLP',
            'reference': ref_m,
            'delta_s': float(np.mean(diff)),
            'point_low': float(np.quantile(boot, 0.025)),
            'point_high': float(np.quantile(boot, 0.975)),
            'family_low': float(np.quantile(boot, 0.025 / 5.0)),
            'family_high': float(np.quantile(boot, 1.0 - 0.025 / 5.0))
        })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(DIAG_DIR / "paired_comparisons.csv", index=False)
    print(paired_df.to_string(index=False))

    # -------------------------------------------------------------
    # 3. GROUPED PERMUTATION ON OCTOBER VAL (TABLE VII)
    # -------------------------------------------------------------
    print("\n--- 3. GROUPED PERMUTATION ON OCTOBER VAL F3 ---")
    GROUPS = {
        "Raw coordinates": ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
        "Zones / geometry": ["PULocationID", "DOLocationID", "haversine_km", "same_zone", "is_airport", "pu_borough_enc", "do_borough_enc"],
        "OSRM routes": ["osrm_km", "osrm_min", "num_steps", "num_nodes", "highway_ratio", "primary_ratio", "local_ratio"],
        "Calendar": ["hour", "weekday", "month", "is_weekend", "is_holiday", "sin_hour", "cos_hour", "sin_dow", "cos_dow", "slot"],
        "History": ["te_od", "te_od_hour", "te_od_slot"]
    }

    # 31-day block bootstrap on October VAL (10,000 trips)
    day_val = pd.factorize(val_df['pickup_datetime'].dt.normalize(), sort=True)[0]
    D_val = int(day_val.max()) + 1
    assert D_val == 31, f"Expected 31 October days, got {D_val}"

    count_val = np.bincount(day_val, minlength=D_val)
    draw_val = np.random.default_rng(20261024).integers(0, D_val, size=(5000, D_val))
    den_val = count_val[draw_val].sum(axis=1)

    rng_perm = np.random.default_rng(20261023)
    perms = [rng_perm.permutation(len(val_df)) for _ in range(10)]

    # Load fitted checkpoints for F3 and compute float64 baseline errors directly
    loaded_models = {}
    base_val_errs = {}
    for m in MODELS:
        loaded_models[m] = []
        base_val_errs[m] = []
        for s in SEEDS:
            ck_file = CK_DIR / f"refit_F3__{m}__{s}.pkl"
            with open(ck_file, 'rb') as f:
                predictor = cloudpickle.load(f)['predictor']
            loaded_models[m].append((s, predictor))
            b_pred = np.asarray(predictor.predict(val_df), dtype=np.float64)
            base_val_errs[m].append(np.abs(b_pred - truth_val_min) * 60.0)

    perm_rows = []
    t_start = time.time()
    for m in MODELS:
        print(f"Evaluating Grouped Permutation across 5 groups for {m}...")
        for g_name, g_cols in GROUPS.items():
            deltas_per_seed = []
            scores = []
            for idx_s, (s, predictor) in enumerate(loaded_models[m]):
                base_err = base_val_errs[m][idx_s]
                accum = []
                for j, p_arr in enumerate(perms):
                    perm_val_df = val_df.copy()
                    perm_val_df.loc[:, g_cols] = val_df.iloc[p_arr][g_cols].to_numpy()
                    p_pred = np.asarray(predictor.predict(perm_val_df), dtype=np.float64)
                    e = np.abs(p_pred - truth_val_min) * 60.0 - base_err
                    accum.append(e)
                    scores.append((s, j, float(e.mean())))
                deltas_per_seed.append(np.mean(accum, axis=0))

            mean_diff = np.mean(deltas_per_seed, axis=0)
            daily_sum = np.bincount(day_val, weights=mean_diff, minlength=D_val)
            boot = daily_sum[draw_val].sum(axis=1) / den_val
            byperm = pd.DataFrame(scores, columns=['seed', 'perm', 'score']).groupby('perm')['score'].mean()

            perm_rows.append({
                'model': m,
                'group': g_name,
                'delta_s': float(mean_diff.mean()),
                'point_low': float(np.quantile(boot, 0.025)),
                'point_high': float(np.quantile(boot, 0.975)),
                'family_low': float(np.quantile(boot, 0.025 / 20.0)),
                'family_high': float(np.quantile(boot, 1.0 - 0.025 / 20.0)),
                'permutation_SD_s': float(byperm.std(ddof=1))
            })

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(DIAG_DIR / "grouped_permutation.csv", index=False)
    print(f"Grouped Permutation finished in {time.time()-t_start:.1f}s.")
    print(perm_df.to_string(index=False))

    # -------------------------------------------------------------
    # 4. SCALING BLOCK BOOTSTRAP CONTRASTS ON TEST (61 DAYS)
    # -------------------------------------------------------------
    scaling_pred_dir = WORKSPACE / "results/scaling/predictions"
    if scaling_pred_dir.exists():
        print("\n--- 4. SCALING 61-DAY BLOCK BOOTSTRAP CONTRASTS ---")
        def load_scale_losses_s(tier_lbl: str, model_name: str) -> np.ndarray:
            per_seed = []
            for s in SEEDS:
                if tier_lbl == '90k':
                    p_npz = PRED_DIR / f"F3__{model_name}__{s}.npz"
                else:
                    p_npz = scaling_pred_dir / f"{model_name.lower()}_{tier_lbl}__seed_{s}.npz"
                with np.load(p_npz) as d:
                    pred = np.asarray(d['test_prediction_minutes'], dtype=np.float64)
                per_seed.append(np.abs(pred - truth_test_min) * 60.0)
            return np.mean(np.stack(per_seed, axis=0), axis=0)

        scale_boot_rows = []
        # Within-model scaling gains: 720k vs 90k and 1080k vs 90k (4 models -> Bonferroni /4 per family)
        for target_tier in ['720k', '1080k']:
            for m in MODELS:
                diff = load_scale_losses_s(target_tier, m) - load_scale_losses_s('90k', m)
                daily_sum = np.bincount(pickup_day_test, weights=diff, minlength=n_days_test)
                boot = daily_sum[draws_test].sum(axis=1) / den_test
                scale_boot_rows.append({
                    'contrast_family': f'{target_tier}_minus_90k',
                    'candidate': f'{m}_{target_tier}',
                    'reference': f'{m}_90k',
                    'delta_s': float(np.mean(diff)),
                    'point_low': float(np.quantile(boot, 0.025)),
                    'point_high': float(np.quantile(boot, 0.975)),
                    'family_low': float(np.quantile(boot, 0.025 / 4.0)),
                    'family_high': float(np.quantile(boot, 1.0 - 0.025 / 4.0))
                })
        # Cross-model comparisons at 720k optimal neural tier (RealMLP vs 3 competitors -> Bonferroni /3)
        rmlp_720 = load_scale_losses_s('720k', 'RealMLP')
        for ref_m in ['LightGBM', 'CatBoost', 'TabM']:
            diff = rmlp_720 - load_scale_losses_s('720k', ref_m)
            daily_sum = np.bincount(pickup_day_test, weights=diff, minlength=n_days_test)
            boot = daily_sum[draws_test].sum(axis=1) / den_test
            scale_boot_rows.append({
                'contrast_family': 'RealMLP_vs_peers_at_720k',
                'candidate': 'RealMLP_720k',
                'reference': f'{ref_m}_720k',
                'delta_s': float(np.mean(diff)),
                'point_low': float(np.quantile(boot, 0.025)),
                'point_high': float(np.quantile(boot, 0.975)),
                'family_low': float(np.quantile(boot, 0.025 / 3.0)),
                'family_high': float(np.quantile(boot, 1.0 - 0.025 / 3.0))
            })
        scale_boot_df = pd.DataFrame(scale_boot_rows)
        scale_boot_df.to_csv(DIAG_DIR / "scaling_block_bootstrap.csv", index=False)
        print(scale_boot_df.to_string(index=False))

    print(f"\n[DONE] All diagnostics saved to {DIAG_DIR}")

if __name__ == '__main__':
    main()
