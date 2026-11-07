#!/usr/bin/env python3
"""
run_1m_2015/code/phase1_sample_azure.py
Extracts 1,110,000 100% fresh, real, non-duplicated trips from Azure Open Datasets NYC TLC 2015.
- TRAIN (M1..M9): 120,000 trips/month = 1,080,000 trips
- VAL (M10): 10,000 trips
- TEST (M11..M12): 10,000 trips/month = 20,000 trips
High-Performance Version:
- Vectorized DuckDB exact month & year 2015 filtering
- Vectorized STRtree spatial zone lookup (<1s per month)
- Fast zip tuple matching for blacklist exclusion (<1.5s per month)
- Vectorized cumcount stratification (<0.1s per month)
- Automatic Gate Pass verification upon completion.
"""

import sys
import time
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import duckdb
import geopandas as gpd
import shapely
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("d:/Download/DAP391m/run_1m_2015/data/2015/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SHP_PATH = Path("d:/Download/DAP391m/run_1m_2015/data/zones/unzipped/taxi_zones/taxi_zones.shp")

# 1. Load Old 120,000 Blacklist
print("Loading Old 120k Blacklist...")
old_ids = set()
old_tuples = set()
dedup_cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

for s in ['train', 'val', 'test']:
    old_p = Path(f"independent_run_map2015/data/2015/processed/{s}.parquet")
    odf = pd.read_parquet(old_p)
    old_ids.update(odf['trip_id'].astype(str))
    for row in zip(odf['pickup_datetime'], odf['pickup_longitude'], odf['pickup_latitude'], odf['dropoff_longitude'], odf['dropoff_latitude']):
        old_tuples.add(row)

print(f"Total Blacklisted: {len(old_ids):,} old trip_ids, {len(old_tuples):,} old 5-tuples.")

# 2. Load TLC Taxi Zones for spatial validation using STRtree
print(f"Loading TLC Taxi Zones shapefile from {SHP_PATH}...")
zones_gdf = gpd.read_file(SHP_PATH).to_crs(epsg=4326)
valid_zone_ids = set(zones_gdf['LocationID'].astype(int))
zone_geoms = zones_gdf.geometry.to_numpy()
zone_loc_ids = zones_gdf['LocationID'].to_numpy()
zone_boroughs = zones_gdf['borough'].to_numpy()
tree = shapely.STRtree(zone_geoms)
print(f"Loaded {len(zones_gdf)} zones into STRtree spatial index.")

def get_azure_blobs_for_month(month: int):
    list_url = f"https://azureopendatastorage.blob.core.windows.net/nyctlc?restype=container&comp=list&prefix=yellow/puYear=2015/puMonth={month}/"
    with urllib.request.urlopen(list_url) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    blobs = [b.find('Name').text for b in root.findall('.//Blob') if b.find('Name').text.endswith('.parquet')]
    return sorted(blobs)

def fast_spatial_zone_lookup(lons, lats):
    pts = shapely.points(lons, lats)
    idx_pts, idx_polys = tree.query(pts, predicate='within')
    
    # Keep first match for boundary points
    df_matches = pd.DataFrame({'pt': idx_pts, 'poly': idx_polys}).drop_duplicates('pt')
    
    loc_res = np.full(len(lons), -1, dtype=np.int32)
    bor_res = np.empty(len(lons), dtype=object)
    
    if len(df_matches) > 0:
        loc_res[df_matches['pt'].to_numpy()] = zone_loc_ids[df_matches['poly'].to_numpy()]
        bor_res[df_matches['pt'].to_numpy()] = zone_boroughs[df_matches['poly'].to_numpy()]
        
    return loc_res, bor_res

def compute_sha256_rank(s_series):
    return [int(hashlib.sha256(f"{s}_seed42".encode()).hexdigest()[:16], 16) for s in s_series]

TARGET_QUOTAS = {
    1: 120000, 2: 120000, 3: 120000, 4: 120000, 5: 120000,
    6: 120000, 7: 120000, 8: 120000, 9: 120000,
    10: 10000,
    11: 10000, 12: 10000
}

con = duckdb.connect()
con.execute('INSTALL httpfs; LOAD httpfs;')

sampled_by_month = {}

for month, quota in TARGET_QUOTAS.items():
    t0 = time.time()
    out_m_file = DATA_DIR / f"sampled_month_{month:02d}.parquet"
    
    # Check if existing cached file is 100% clean
    if out_m_file.exists():
        m_df = pd.read_parquet(out_m_file)
        if len(m_df) == quota and (m_df['pickup_datetime'].dt.month == month).all() and (m_df['pickup_datetime'].dt.year == 2015).all():
            print(f"Month {month:02d} already verified clean ({len(m_df):,} trips). Loaded from cache.")
            sampled_by_month[month] = m_df
            continue

    print(f"\n=======================================================")
    print(f"SAMPLING MONTH {month:02d}/2015 (Target Quota: {quota:,} trips)")
    print(f"=======================================================")
    
    blobs = get_azure_blobs_for_month(month)
    print(f"Found {len(blobs)} Azure parquet part files.")

    candidates = []
    total_candidates = 0
    target_pool = max(quota * 2, 250000)

    for blob_idx, blob_name in enumerate(blobs):
        file_url = f"https://azureopendatastorage.blob.core.windows.net/nyctlc/{blob_name}"
        fname = Path(blob_name).name
        print(f"  Reading part {blob_idx+1}/{len(blobs)}: {fname}...")
        
        # Read with EXACT month & year filtering
        q = f"""
        SELECT 
            '{blob_name}' as blob_file,
            row_number() over() - 1 as row_idx,
            vendorID as VendorID,
            tpepPickupDateTime as pickup_datetime,
            tpepDropoffDateTime as dropoff_datetime,
            passengerCount as passenger_count,
            tripDistance as trip_distance,
            rateCodeId as rate_code,
            startLon as pickup_longitude,
            startLat as pickup_latitude,
            endLon as dropoff_longitude,
            endLat as dropoff_latitude
        FROM '{file_url}'
        WHERE 
            month(tpepPickupDateTime) = {month} AND year(tpepPickupDateTime) = 2015
            AND startLon IS NOT NULL AND startLat IS NOT NULL 
            AND endLon IS NOT NULL AND endLat IS NOT NULL
            AND startLon BETWEEN -74.5 AND -73.5
            AND startLat BETWEEN 40.4 AND 41.2
            AND endLon BETWEEN -74.5 AND -73.5
            AND endLat BETWEEN 40.4 AND 41.2
            AND rateCodeId IN (1, 2)
            AND passengerCount > 0
            AND tripDistance >= 0.0
            AND tpepDropoffDateTime > tpepPickupDateTime
        """
        try:
            pdf = con.execute(q).fetchdf()
        except Exception as e:
            print(f"    Warning reading {fname}: {e}")
            continue

        if len(pdf) == 0:
            continue

        # Compute duration in minutes
        pdf['dur_min'] = (pdf['dropoff_datetime'] - pdf['pickup_datetime']).dt.total_seconds() / 60.0
        
        # Eligibility filters
        # 1. 1 to 180 min
        pdf = pdf[(pdf['dur_min'] >= 1.0) & (pdf['dur_min'] <= 180.0)].copy()
        
        # 2. Haversine distance & speed <= 60 mph
        lat1, lon1 = np.radians(pdf['pickup_latitude']), np.radians(pdf['pickup_longitude'])
        lat2, lon2 = np.radians(pdf['dropoff_latitude']), np.radians(pdf['dropoff_longitude'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        pdf['haversine_km'] = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        
        # Speed checks (<= 60 mph = 96.56 km/h)
        speed_recorded_kmh = (pdf['trip_distance'] * 1.60934) / (pdf['dur_min'] / 60.0)
        speed_hav_kmh = pdf['haversine_km'] / (pdf['dur_min'] / 60.0)
        pdf = pdf[(speed_recorded_kmh <= 96.56) & (speed_hav_kmh <= 96.56)].copy()
        
        # Zero-distance trips longer than 2 minutes excluded
        pdf = pdf[~((pdf['trip_distance'] == 0.0) & (pdf['dur_min'] > 2.0))].copy()
        
        # Exclude month boundary trips
        if month == 9:
            pdf = pdf[pdf['dropoff_datetime'] < '2015-10-01'].copy()
        elif month == 10:
            pdf = pdf[pdf['dropoff_datetime'] < '2015-11-01'].copy()
        elif month == 12:
            pdf = pdf[pdf['dropoff_datetime'] < '2016-01-01'].copy()

        # Construct trip_id
        pdf['trip_id'] = [
            f"2015|month={month:02d}/{row.blob_file.split('/')[-1]}|{row.row_idx}"
            for row in pdf.itertuples()
        ]

        # Blacklist exclusion (trip_id & fast zip 5-tuple)
        pdf = pdf[~pdf['trip_id'].isin(old_ids)].copy()
        
        t_tuples = list(zip(pdf['pickup_datetime'], pdf['pickup_longitude'], pdf['pickup_latitude'], pdf['dropoff_longitude'], pdf['dropoff_latitude']))
        is_blacklisted = [t in old_tuples for t in t_tuples]
        pdf = pdf[~np.array(is_blacklisted)].copy()

        # Deduplicate internally
        pdf = pdf.drop_duplicates(subset=dedup_cols)
        pdf = pdf.drop_duplicates(subset=['trip_id'])

        candidates.append(pdf)
        total_candidates += len(pdf)
        print(f"    Accepted {len(pdf):,} valid clean candidates (cumulative: {total_candidates:,}/{target_pool:,})")

        if total_candidates >= target_pool:
            break

    cand_df = pd.concat(candidates, ignore_index=True)
    cand_df = cand_df.drop_duplicates(subset=dedup_cols)
    cand_df = cand_df.drop_duplicates(subset=['trip_id'])
    print(f"Total clean candidates in Month {month:02d}: {len(cand_df):,}")
    assert len(cand_df) >= quota, f"Not enough clean candidates! Needed {quota}, got {len(cand_df)}"

    # Vectorized STRtree spatial zone lookup
    t_sp = time.time()
    pu_locs, pu_bors = fast_spatial_zone_lookup(cand_df['pickup_longitude'].to_numpy(), cand_df['pickup_latitude'].to_numpy())
    do_locs, do_bors = fast_spatial_zone_lookup(cand_df['dropoff_longitude'].to_numpy(), cand_df['dropoff_latitude'].to_numpy())
    print(f"Fast spatial lookup completed in {time.time()-t_sp:.2f}s")

    cand_df['PULocationID'] = pu_locs
    cand_df['DOLocationID'] = do_locs
    cand_df['pu_borough'] = pu_bors
    cand_df['do_borough'] = do_bors

    # Filter out points that fell outside valid taxi zone polygons (-1)
    valid_mask = (cand_df['PULocationID'] > 0) & (cand_df['DOLocationID'] > 0)
    cand_df = cand_df[valid_mask].copy()
    cand_df['PULocationID'] = cand_df['PULocationID'].astype('int32')
    cand_df['DOLocationID'] = cand_df['DOLocationID'].astype('int32')
    print(f"Candidates with valid zones: {len(cand_df):,}")

    # Compute corridor group
    borough_map = {'Bronx': 0, 'Brooklyn': 1, 'EWR': 2, 'Manhattan': 3, 'Queens': 4, 'Staten Island': 5}
    cand_df['pu_borough_enc'] = cand_df['pu_borough'].map(borough_map).fillna(0).astype('int64')
    cand_df['do_borough_enc'] = cand_df['do_borough'].map(borough_map).fillna(0).astype('int64')
    cand_df['is_airport'] = (cand_df['PULocationID'].isin([1, 132, 138]) | cand_df['DOLocationID'].isin([1, 132, 138])).astype('int64')

    c1 = cand_df['is_airport'] == 1
    c2 = (~c1) & (cand_df['pu_borough_enc'] == 3) & (cand_df['do_borough_enc'] == 3)
    c3 = (~c1) & (cand_df['pu_borough_enc'] == 3) & (cand_df['do_borough_enc'] != 3)
    c4 = (~c1) & (cand_df['pu_borough_enc'] != 3) & (cand_df['do_borough_enc'] == 3)
    c5 = (~c1) & (cand_df['pu_borough_enc'] != 3) & (cand_df['do_borough_enc'] != 3)
    
    corridor = np.zeros(len(cand_df), dtype='int64')
    corridor[c1] = 1
    corridor[c2] = 2
    corridor[c3] = 3
    corridor[c4] = 4
    corridor[c5] = 5
    cand_df['corridor'] = corridor

    # Stratified sampling
    cand_df['pickup_date'] = cand_df['pickup_datetime'].dt.strftime('%Y-%m-%d')
    cand_df['hour'] = cand_df['pickup_datetime'].dt.hour
    
    if month <= 9:
        strata_cols = ['pickup_date', 'hour', 'corridor']
    else:
        strata_cols = ['pickup_date', 'hour']

    # Deterministic SHA-256 rank within each candidate
    t_sha = time.time()
    cand_df['sha_rank'] = compute_sha256_rank(cand_df['trip_id'])
    print(f"SHA-256 ranking calculated in {time.time()-t_sha:.2f}s")

    # Largest remainder proportional quota with min 1
    strata_counts = cand_df.groupby(strata_cols).size()
    total_cand = len(cand_df)
    shares = (strata_counts / total_cand) * quota
    alloc = np.floor(shares).astype(int)
    alloc = np.maximum(alloc, 1)
    
    # Adjust remainder
    rem = quota - alloc.sum()
    if rem > 0:
        fractions = shares - np.floor(shares)
        top_strata = fractions.nlargest(rem).index
        alloc.loc[top_strata] += 1
    elif rem < 0:
        excess = -rem
        top_alloc = alloc.nlargest(excess).index
        alloc.loc[top_alloc] -= 1

    # High-performance Vectorized Stratified Sampling (100x faster than iteration)
    t_sample = time.time()
    df_sorted = cand_df.sort_values(strata_cols + ['sha_rank'])
    df_sorted['rank_in_stratum'] = df_sorted.groupby(strata_cols).cumcount()
    alloc_df = alloc.reset_index(name='n_take')
    merged = df_sorted.merge(alloc_df, on=strata_cols, how='left')
    sampled_df = merged[merged['rank_in_stratum'] < merged['n_take']].drop(columns=['rank_in_stratum', 'n_take'])
    
    if len(sampled_df) > quota:
        sampled_df = sampled_df.sort_values('sha_rank').head(quota)
    elif len(sampled_df) < quota:
        needed = quota - len(sampled_df)
        leftover = cand_df[~cand_df['trip_id'].isin(sampled_df['trip_id'])].sort_values('sha_rank')
        sampled_df = pd.concat([sampled_df, leftover.head(needed)], ignore_index=True)

    print(f"Vectorized stratification completed in {time.time()-t_sample:.2f}s")

    assert len(sampled_df) == quota, f"Failed to get exact quota for month {month}: {len(sampled_df)} vs {quota}"
    assert sampled_df['trip_id'].nunique() == quota, "Internal duplicate detected in sampled cohort!"
    assert (sampled_df['pickup_datetime'].dt.month == month).all(), f"Stray month detected in month {month} sample!"
    assert (sampled_df['pickup_datetime'].dt.year == 2015).all(), f"Stray year detected in month {month} sample!"

    keep_cols = [
        'trip_id', 'pickup_datetime', 'dropoff_datetime', 'dur_min', 'VendorID',
        'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'PULocationID', 'DOLocationID',
        'pu_borough_enc', 'do_borough_enc', 'is_airport', 'haversine_km'
    ]
    sampled_df = sampled_df[keep_cols].copy()
    sampled_df.to_parquet(out_m_file, index=False)
    print(f"Month {month:02d} successfully sampled ({len(sampled_df):,} trips) in {time.time()-t0:.1f}s -> {out_m_file}")
    sampled_by_month[month] = sampled_df

# Combine into TRAIN (1..9), VAL (10), TEST (11..12)
print("\nCombining monthly samples into outer splits...")
train_list = [sampled_by_month[m] for m in range(1, 10)]
raw_sampled_train = pd.concat(train_list, ignore_index=True)

raw_sampled_val = sampled_by_month[10].copy()

test_list = [sampled_by_month[11], sampled_by_month[12]]
raw_sampled_test = pd.concat(test_list, ignore_index=True)

raw_sampled_train.to_parquet(DATA_DIR / "raw_sampled_train.parquet", index=False)
raw_sampled_val.to_parquet(DATA_DIR / "raw_sampled_val.parquet", index=False)
raw_sampled_test.to_parquet(DATA_DIR / "raw_sampled_test.parquet", index=False)

print(f"TRAIN: {len(raw_sampled_train):,} rows -> {DATA_DIR / 'raw_sampled_train.parquet'}")
print(f"VAL:   {len(raw_sampled_val):,} rows   -> {DATA_DIR / 'raw_sampled_val.parquet'}")
print(f"TEST:  {len(raw_sampled_test):,} rows  -> {DATA_DIR / 'raw_sampled_test.parquet'}")
print("PHASE 1 SAMPLING COMPLETED SUCCESSFULLY!")
