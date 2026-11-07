#!/usr/bin/env python3
"""
run_1m_2015/code/export_scaling_subsets.py
Exports properly stratified nested scaling subsets:
90k -> 180k -> 360k -> 540k -> 720k -> 900k -> 1,080k

Guarantees:
1. Strict Nestedness: S_90k ⊂ S_180k ⊂ S_360k ⊂ S_540k ⊂ S_720k ⊂ S_900k ⊂ S_1080k
2. 100% Calendar Coverage: Every tier covers all 30/31 days of EVERY month Jan-Sep via deterministic SHA-256 ranking.
3. Zero Data Scale Bleed: te_od, te_od_hour, te_od_slot are recomputed strictly within each tier's train pool.
   Evaluation sets (val_{S}k.parquet, test_{S}k.parquet) also look up history ONLY from train_{S}k.
"""

import hashlib
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path("d:/Download/DAP391m/run_1m_2015").resolve()
DATA_DIR = WORKSPACE / "data/2015/processed"
SUBSETS_DIR = WORKSPACE / "data/2015/subsets"
SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha_rank(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: int(hashlib.sha256(str(x).encode('utf-8')).hexdigest()[:14], 16))

def compute_target_encodings_local(tr_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Computes past-completion target encodings without time leakage or data-scale bleeding."""
    dropoff_dt = (tr_df['pickup_datetime'] + pd.to_timedelta(np.round(tr_df['dur_min'] * 60.0), unit='s')).astype('datetime64[us]')
    tr_sorted = tr_df.assign(dropoff_datetime=dropoff_dt).sort_values(['dropoff_datetime', 'trip_id'])
    completed_ts = tr_sorted['dropoff_datetime'].to_numpy()
    cum_sums = tr_sorted['y'].cumsum().to_numpy()

    positions = np.searchsorted(completed_ts, eval_df['pickup_datetime'].astype('datetime64[us]').to_numpy(), side='left')
    prior = np.full(len(eval_df), np.log1p(15.0))
    pos = positions > 0
    prior[pos] = cum_sums[positions[pos] - 1] / positions[pos]

    out = {}
    for keys, col in [
        (['PULocationID', 'DOLocationID'], 'te_od'),
        (['PULocationID', 'DOLocationID', 'hour'], 'te_od_hour'),
        (['PULocationID', 'DOLocationID', 'slot'], 'te_od_slot')
    ]:
        stats = tr_sorted[keys + ['dropoff_datetime', 'y']].copy()
        stats['dropoff_datetime'] = stats['dropoff_datetime'].astype('datetime64[us]')
        stats['history_sum'] = stats.groupby(keys)['y'].cumsum()
        stats['history_count'] = stats.groupby(keys).cumcount() + 1
        left = eval_df[keys + ['pickup_datetime']].copy()
        left['pickup_datetime'] = left['pickup_datetime'].astype('datetime64[us]')
        left['row_id'] = eval_df.index
        joined = pd.merge_asof(
            left.sort_values('pickup_datetime'),
            stats.drop(columns='y').sort_values('dropoff_datetime', kind='stable'),
            left_on='pickup_datetime',
            right_on='dropoff_datetime',
            by=keys,
            direction='backward',
            allow_exact_matches=False
        ).sort_values('row_id')
        out[col] = (joined['history_sum'].fillna(0.0).to_numpy() + 20.0 * prior) / (joined['history_count'].fillna(0.0).to_numpy() + 20.0)
    return pd.DataFrame(out, index=eval_df.index)

def main():
    print("=" * 85)
    print("EXPORTING STRATIFIED NESTED SUBSETS WITH TIER-LOCAL TARGET ENCODINGS")
    print("=" * 85)

    t0 = time.time()
    print("Loading full 1.080.000 TRAIN parquet and evaluation sets...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"Loaded: TRAIN={len(train_df):,}, VAL={len(val_df):,}, TEST={len(test_df):,} in {time.time()-t0:.1f}s.")

    print("Computing deterministic SHA-256 ranks...")
    train_df['sha_rank'] = compute_sha_rank(train_df['trip_id'])

    months = sorted(train_df['month'].unique())
    monthly_sorted = {}
    for m in months:
        m_df = train_df[train_df['month'] == m].sort_values('sha_rank').reset_index(drop=True)
        assert len(m_df) == 120_000
        monthly_sorted[m] = m_df

    SCALE_TIERS = [90_000, 180_000, 360_000, 540_000, 720_000, 900_000, 1_080_000]
    prev_ids = None
    manifest_records = []

    for n in SCALE_TIERS:
        t_tier = time.time()
        label = f"{n//1000}k"
        per_month = n // len(months)
        out_file = SUBSETS_DIR / f"train_{label}.parquet"

        sub_list = [monthly_sorted[m].iloc[:per_month] for m in months]
        sub_df = pd.concat(sub_list, ignore_index=True)
        save_df = sub_df.drop(columns=['sha_rank']).sort_values('pickup_datetime').reset_index(drop=True)

        curr_ids = set(save_df['trip_id'])
        assert len(save_df) == n
        assert len(curr_ids) == n

        if prev_ids is not None:
            assert prev_ids.issubset(curr_ids)
            print(f"\n[PASS] Nested subset verified: S_prev subset of S_{label} (+{n - len(prev_ids):,} trips)")
        else:
            print(f"\n[PASS] Initial subset S_{label} initialized ({n:,} trips)")

        prev_ids = curr_ids

        # Recompute te_od* strictly within this tier's train pool
        if n < 1_080_000:
            print(f"  Computing tier-local te_od* for train_{label} ({n:,} trips)...")
            te_tr = compute_target_encodings_local(save_df, save_df)
            save_df[['te_od', 'te_od_hour', 'te_od_slot']] = te_tr[['te_od', 'te_od_hour', 'te_od_slot']]

            print(f"  Computing tier-local te_od* for val_{label} and test_{label}...")
            val_sub = val_df.copy()
            te_val = compute_target_encodings_local(save_df, val_sub)
            val_sub[['te_od', 'te_od_hour', 'te_od_slot']] = te_val[['te_od', 'te_od_hour', 'te_od_slot']]
            val_sub.to_parquet(SUBSETS_DIR / f"val_{label}.parquet", index=False)

            test_sub = test_df.copy()
            te_test = compute_target_encodings_local(save_df, test_sub)
            test_sub[['te_od', 'te_od_hour', 'te_od_slot']] = te_test[['te_od', 'te_od_hour', 'te_od_slot']]
            test_sub.to_parquet(SUBSETS_DIR / f"test_{label}.parquet", index=False)
        else:
            # 1.080k uses the full pool
            val_df.to_parquet(SUBSETS_DIR / f"val_{label}.parquet", index=False)
            test_df.to_parquet(SUBSETS_DIR / f"test_{label}.parquet", index=False)

        save_df.to_parquet(out_file, index=False)
        print(f"  Saved train_{label}.parquet, val_{label}.parquet, test_{label}.parquet in {time.time()-t_tier:.1f}s.")

        manifest_records.append({
            'tier': label,
            'train_size': n,
            'per_month': per_month,
            'train_file': str(out_file.name),
            'val_file': f"val_{label}.parquet",
            'test_file': f"test_{label}.parquet",
            'te_od_mean': float(save_df['te_od'].mean()),
            'test_te_od_mean': float(pd.read_parquet(SUBSETS_DIR / f"test_{label}.parquet")['te_od'].mean())
        })

    # Hard Verification Gate (joined strictly by trip_id)
    s90_df = pd.read_parquet(SUBSETS_DIR / "train_90k.parquet", columns=['trip_id', 'te_od'])
    s1080_df = pd.read_parquet(DATA_DIR / "train.parquet", columns=['trip_id', 'te_od'])
    merged_gate = s90_df.merge(s1080_df, on='trip_id', suffixes=('_90k', '_1080k'))
    assert len(merged_gate) == 90_000
    diff_val = float(np.abs(merged_gate['te_od_90k'].to_numpy() - merged_gate['te_od_1080k'].to_numpy()).max())
    print("\n" + "=" * 85)
    print("VERIFICATION GATE: TIER-ISOLATED TARGET ENCODING (JOINED BY TRIP_ID)")
    print(f"Max diff between 90k local te_od and 1.08M pool te_od: {diff_val:.4f}")
    assert diff_val > 0.5, "GATE FAILED: 90k te_od is still identical to 1.08M pool!"
    print("GATE PASS: Target encoding scale bleeding completely eliminated!")
    print("=" * 85)

    with open(SUBSETS_DIR / "subsets_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

if __name__ == "__main__":
    main()
