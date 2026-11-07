#!/usr/bin/env python3
"""
run_1m_2015/code/phase2_extract_features.py
Computes all 34 canonical features across 1,110,000 trips:
- TRAIN: 1,080,000 trips
- VAL:      10,000 trips
- TEST:     20,000 trips

Steps:
1. Parallel OSRM routing against local daemon (http://127.0.0.1:5001) via aiohttp.
   Extracts: route_status, osrm_km, osrm_min, num_steps, num_nodes, highway_ratio, primary_ratio, local_ratio.
2. 34 Features (F0, F1, F2, F3) + label y (float64).
3. Strictly non-leaking past-completion target encodings (te_od, te_od_hour, te_od_slot) with smoothing 20 and prior log1p(15).
4. Saves train.parquet, val.parquet, test.parquet, and manifest.json with SHA-256 checksums.
"""

import sys
import time
import json
import hashlib
import asyncio
from pathlib import Path
import aiohttp
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path("d:/Download/DAP391m/run_1m_2015").resolve()
DATA_DIR = WORKSPACE / "data/2015/processed"
CACHE_DIR = WORKSPACE / "data/osrm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OSRM_URL = "http://127.0.0.1:5001/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false&steps=true&annotations=nodes,distance,duration"

PRIMARY_KEYWORDS = (
    'avenue', 'ave', 'parkway', 'boulevard', 'blvd', 'expressway',
    'expy', 'broadway', 'bowery', 'square', 'concourse', 'turnpike'
)

HOLIDAYS_2015 = {
    '2015-01-01', '2015-01-19', '2015-02-16', '2015-05-25', '2015-07-03', '2015-07-04',
    '2015-09-07', '2015-10-12', '2015-11-11', '2015-11-26', '2015-12-25'
}

FEATURE_GROUPS = {
    "F0": [
        "VendorID", "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "pickup_timestamp"
    ],
    "F1": [
        "VendorID", "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "pickup_timestamp",
        "PULocationID", "DOLocationID", "haversine_km", "same_zone", "is_airport",
        "pu_borough_enc", "do_borough_enc",
        "osrm_km", "osrm_min", "num_steps", "num_nodes",
        "highway_ratio", "primary_ratio", "local_ratio"
    ],
    "F2": [
        "VendorID", "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "pickup_timestamp",
        "hour", "weekday", "month", "is_weekend", "is_holiday",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow", "slot",
        "te_od", "te_od_hour", "te_od_slot"
    ],
    "F3": [
        "VendorID", "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "pickup_timestamp",
        "PULocationID", "DOLocationID", "haversine_km", "same_zone", "is_airport",
        "pu_borough_enc", "do_borough_enc",
        "osrm_km", "osrm_min", "num_steps", "num_nodes",
        "highway_ratio", "primary_ratio", "local_ratio",
        "hour", "weekday", "month", "is_weekend", "is_holiday",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow", "slot",
        "te_od", "te_od_hour", "te_od_slot"
    ]
}

def parse_osrm_json(data, hav_km):
    if not data or data.get('code') != 'Ok' or not data.get('routes'):
        return {
            'route_status': 'snap_over_1km',
            'osrm_km': np.nan, 'osrm_min': np.nan,
            'num_steps': np.nan, 'num_nodes': np.nan,
            'highway_ratio': np.nan, 'primary_ratio': np.nan, 'local_ratio': np.nan
        }

    # Check waypoints snapping distance
    waypoints = data.get('waypoints', [])
    for wp in waypoints:
        if wp.get('distance', 0.0) > 1000.0:
            return {
                'route_status': 'snap_over_1km',
                'osrm_km': np.nan, 'osrm_min': np.nan,
                'num_steps': np.nan, 'num_nodes': np.nan,
                'highway_ratio': np.nan, 'primary_ratio': np.nan, 'local_ratio': np.nan
            }

    rt = data['routes'][0]
    total_dist = rt.get('distance', 0.0)
    total_dur = rt.get('duration', 0.0)
    legs = rt.get('legs', [])
    if not legs:
        return {
            'route_status': 'snap_over_1km',
            'osrm_km': np.nan, 'osrm_min': np.nan,
            'num_steps': np.nan, 'num_nodes': np.nan,
            'highway_ratio': np.nan, 'primary_ratio': np.nan, 'local_ratio': np.nan
        }

    leg = legs[0]
    steps = leg.get('steps', [])
    nodes = leg.get('annotation', {}).get('nodes', [])

    if total_dist == 0.0 or hav_km == 0.0:
        return {
            'route_status': 'zero_distance',
            'osrm_km': 0.0, 'osrm_min': 0.0,
            'num_steps': 2.0, 'num_nodes': 2.0,
            'highway_ratio': 0.0, 'primary_ratio': 0.0, 'local_ratio': 0.0
        }

    d_hw, d_prim, d_loc = 0.0, 0.0, 0.0
    for s in steps:
        sd = s.get('distance', 0.0)
        dur = s.get('duration', 0.0)
        spd = (sd / dur * 3.6) if dur > 0 else 0.0
        cls = [c for i in s.get('intersections', []) if 'classes' in i for c in i.get('classes', [])]
        nm = s.get('name', '').lower()

        if 'motorway' in cls or spd >= 70.0:
            d_hw += sd
        elif any(k in nm for k in PRIMARY_KEYWORDS) or spd > 25.5:
            d_prim += sd
        else:
            d_loc += sd

    return {
        'route_status': 'ok',
        'osrm_km': total_dist / 1000.0,
        'osrm_min': total_dur / 60.0,
        'num_steps': float(len(steps)),
        'num_nodes': float(len(nodes)),
        'highway_ratio': d_hw / total_dist,
        'primary_ratio': d_prim / total_dist,
        'local_ratio': d_loc / total_dist
    }

async def route_batch_async(coords_list, hav_list):
    results = [None] * len(coords_list)
    connector = aiohttp.TCPConnector(limit=64)
    timeout = aiohttp.ClientTimeout(total=5)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def fetch(idx, lon1, lat1, lon2, lat2, hav):
            url = OSRM_URL.format(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results[idx] = parse_osrm_json(data, hav)
                    else:
                        results[idx] = parse_osrm_json(None, hav)
            except Exception:
                results[idx] = parse_osrm_json(None, hav)

        # Process in chunks of 5,000 to manage memory & sockets
        CHUNK_SIZE = 5000
        for i in range(0, len(coords_list), CHUNK_SIZE):
            chunk_coords = coords_list[i:i+CHUNK_SIZE]
            chunk_hav = hav_list[i:i+CHUNK_SIZE]
            tasks = [
                fetch(i + j, c[0], c[1], c[2], c[3], h)
                for j, (c, h) in enumerate(zip(chunk_coords, chunk_hav))
            ]
            await asyncio.gather(*tasks)
            done_cnt = min(i + CHUNK_SIZE, len(coords_list))
            if done_cnt % 50000 == 0 or done_cnt == len(coords_list):
                print(f"    Routed {done_cnt:,} / {len(coords_list):,} trips ({done_cnt/len(coords_list):.1%})...")

    return results

def get_or_compute_osrm(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"osrm_{split_name}.parquet"
    if cache_path.exists():
        osrm_df = pd.read_parquet(cache_path)
        if len(osrm_df) == len(df):
            print(f"[{split_name}] Loaded OSRM features from cache: {cache_path}")
            return osrm_df

    print(f"[{split_name}] Routing {len(df):,} trips against local OSRM daemon...")
    t0 = time.time()
    coords = list(zip(
        df['pickup_longitude'].to_numpy(), df['pickup_latitude'].to_numpy(),
        df['dropoff_longitude'].to_numpy(), df['dropoff_latitude'].to_numpy()
    ))
    havs = df['haversine_km'].to_numpy()

    res_list = asyncio.run(route_batch_async(coords, havs))
    osrm_df = pd.DataFrame(res_list)
    osrm_df['trip_id'] = df['trip_id'].to_numpy()
    osrm_df.to_parquet(cache_path, index=False)
    print(f"[{split_name}] OSRM routing finished in {time.time()-t0:.1f}s ({(len(df)/(time.time()-t0)):.0f} req/s) -> {cache_path}")
    return osrm_df

def compute_target_encodings(tr_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    print(f"Computing non-leaking past-completion target encodings for {len(eval_df):,} trips...")
    dropoff_dt = (tr_df['pickup_datetime'] + pd.to_timedelta(np.round(tr_df['dur_min'] * 60.0), unit='s')).astype('datetime64[us]')
    tr_sorted = tr_df.assign(dropoff_datetime=dropoff_dt).sort_values(['dropoff_datetime', 'trip_id'])
    completed_ts = tr_sorted['dropoff_datetime'].to_numpy()
    cum_sums = tr_sorted['y'].cumsum().to_numpy()

    positions = np.searchsorted(completed_ts, eval_df['pickup_datetime'].astype('datetime64[us]').to_numpy(), side='left')
    prior = np.full(len(eval_df), np.log1p(15.0))
    pos = positions > 0
    prior[pos] = cum_sums[positions[pos] - 1] / positions[pos]

    out_df = pd.DataFrame(index=eval_df.index)
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
        out_df[col] = (joined['history_sum'].fillna(0.0).to_numpy() + 20.0 * prior) / (joined['history_count'].fillna(0.0).to_numpy() + 20.0)

    return out_df

def build_split(raw_path: Path, split_name: str, train_ref_df: pd.DataFrame = None):
    print(f"\n=======================================================")
    print(f"BUILDING {split_name.upper()} DATASET (from {raw_path.name})")
    print(f"=======================================================")
    raw_df = pd.read_parquet(raw_path)
    df = pd.DataFrame()
    df['trip_id'] = raw_df['trip_id']
    df['pickup_datetime'] = pd.to_datetime(raw_df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(raw_df['dropoff_datetime'])
    df['dur_min'] = raw_df['dur_min'].astype('float64')
    df['y'] = np.log1p(np.maximum(df['dur_min'].to_numpy(dtype='float64'), 0.0))

    # Base F0
    df['VendorID'] = raw_df['VendorID'].astype('int64')
    df['passenger_count'] = raw_df['passenger_count'].astype('int64')
    df['pickup_longitude'] = raw_df['pickup_longitude'].astype('float64')
    df['pickup_latitude'] = raw_df['pickup_latitude'].astype('float64')
    df['dropoff_longitude'] = raw_df['dropoff_longitude'].astype('float64')
    df['dropoff_latitude'] = raw_df['dropoff_latitude'].astype('float64')
    df['pickup_timestamp'] = (df['pickup_datetime'] - pd.Timestamp('2015-01-01')).dt.total_seconds() / 86400.0

    # Calendar features
    df['hour'] = df['pickup_datetime'].dt.hour.astype('int64')
    df['weekday'] = df['pickup_datetime'].dt.weekday.astype('int64')
    df['month'] = df['pickup_datetime'].dt.month.astype('int64')
    df['is_weekend'] = (df['weekday'] >= 5).astype('int64')
    df['is_holiday'] = df['pickup_datetime'].dt.strftime('%Y-%m-%d').isin(HOLIDAYS_2015).astype('int64')
    df['sin_hour'] = np.sin(2.0 * np.pi * df['hour'].to_numpy(dtype='float64') / 24.0)
    df['cos_hour'] = np.cos(2.0 * np.pi * df['hour'].to_numpy(dtype='float64') / 24.0)
    df['sin_dow'] = np.sin(2.0 * np.pi * df['weekday'].to_numpy(dtype='float64') / 7.0)
    df['cos_dow'] = np.cos(2.0 * np.pi * df['weekday'].to_numpy(dtype='float64') / 7.0)
    df['slot'] = (df['weekday'] * 24 + df['hour']).astype('int64')

    # Spatial features
    df['PULocationID'] = raw_df['PULocationID'].astype('int64')
    df['DOLocationID'] = raw_df['DOLocationID'].astype('int64')
    df['haversine_km'] = raw_df['haversine_km'].astype('float64')
    df['same_zone'] = (df['PULocationID'] == df['DOLocationID']).astype('int64')
    df['is_airport'] = raw_df['is_airport'].astype('int64')
    df['pu_borough_enc'] = raw_df['pu_borough_enc'].astype('int64')
    df['do_borough_enc'] = raw_df['do_borough_enc'].astype('int64')

    # OSRM features
    osrm_feats = get_or_compute_osrm(df, split_name)
    for col in ['route_status', 'osrm_km', 'osrm_min', 'num_steps', 'num_nodes', 'highway_ratio', 'primary_ratio', 'local_ratio']:
        df[col] = osrm_feats[col].to_numpy()

    # Target encodings
    ref_pool = train_ref_df if train_ref_df is not None else df
    te_feats = compute_target_encodings(ref_pool, df)
    for col in ['te_od', 'te_od_hour', 'te_od_slot']:
        df[col] = te_feats[col].to_numpy()

    out_p = DATA_DIR / f"{split_name}.parquet"
    df.to_parquet(out_p, index=False)
    print(f"[{split_name}] Saved {len(df):,} rows -> {out_p}")
    return df

def main():
    t_start = time.time()
    raw_train = DATA_DIR / "raw_sampled_train.parquet"
    raw_val = DATA_DIR / "raw_sampled_val.parquet"
    raw_test = DATA_DIR / "raw_sampled_test.parquet"

    assert raw_train.exists() and raw_val.exists() and raw_test.exists(), "Raw sampled files missing!"

    # 1. Build TRAIN (1.08M)
    train_df = build_split(raw_train, "train")

    # 2. Build VAL (10k)
    val_df = build_split(raw_val, "val", train_ref_df=train_df)

    # 3. Build TEST (20k)
    test_df = build_split(raw_test, "test", train_ref_df=train_df)

    # 4. Generate manifest.json with SHA-256
    print("\nGenerating manifest.json with SHA-256 checksums...")
    manifest = {
        "dataset_name": "NYC TLC Yellow Taxi 2015 - 1.11M True Fresh Cohort",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": FEATURE_GROUPS,
        "files": {}
    }

    for name in ["train", "val", "test"]:
        p = DATA_DIR / f"{name}.parquet"
        data_bytes = p.read_bytes()
        manifest["files"][name] = {
            "path": str(p.relative_to(WORKSPACE)),
            "rows": len(pd.read_parquet(p)),
            "sha256": hashlib.sha256(data_bytes).hexdigest()
        }

    manifest_p = DATA_DIR / "manifest.json"
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_p}")
    print(f"PHASE 2 COMPLETED SUCCESSFULLY in {time.time()-t_start:.1f}s!")

if __name__ == "__main__":
    main()
