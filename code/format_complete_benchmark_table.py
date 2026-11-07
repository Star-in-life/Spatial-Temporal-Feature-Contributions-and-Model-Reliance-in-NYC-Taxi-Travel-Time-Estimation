#!/usr/bin/env python3
"""
run_1m_2015/code/format_complete_benchmark_table.py
Formats the definitive Table VI and complete Factorial Ablation benchmark table,
integrating OSRM and TEMP reference baselines into test_ablation_summary.csv
and generating table_vi_complete_benchmark.csv.
"""

import pandas as pd
from pathlib import Path

WORKSPACE = Path("d:/Download/DAP391m/run_1m_2015").resolve()
REFITS_DIR = WORKSPACE / "results/refits"

ablation_df = pd.read_csv(REFITS_DIR / "test_ablation_summary.csv")
osrm_df = pd.read_csv(WORKSPACE / "results/scaling/scaling_osrm_summary.csv")
temp_df = pd.read_csv(WORKSPACE / "results/scaling/scaling_temp_summary.csv")

osrm_90 = osrm_df[osrm_df['tier'] == '90k'].iloc[0]
temp_90 = temp_df[temp_df['tier'] == '90k'].iloc[0]

baseline_rows = [
    {
        'F': 'Baseline',
        'model': 'OSRM',
        'n_seeds': 1,
        'TEST_MAE_s_mean': float(osrm_90['test_mae_s']),
        'TEST_MAE_s_std': 0.0,
        'TEST_RMSE_s_mean': float(osrm_90['test_rmse_s']),
        'TEST_RMSE_s_std': 0.0,
        'TEST_MEDAE_s_mean': float(osrm_90['test_medae_s']),
        'TEST_MEDAE_s_std': 0.0,
        'TEST_WAPE_mean': float(osrm_90['test_wape']),
        'TEST_WAPE_std': 0.0,
        'TEST_RMSLE_mean': float(osrm_90['test_rmsle']),
        'TEST_RMSLE_std': 0.0,
        'TEST_R2_mean': float(osrm_90['test_r2']),
        'TEST_R2_std': 0.0
    },
    {
        'F': 'Baseline',
        'model': 'TEMP',
        'n_seeds': 1,
        'TEST_MAE_s_mean': float(temp_90['test_mae_s']),
        'TEST_MAE_s_std': 0.0,
        'TEST_RMSE_s_mean': float(temp_90['test_rmse_s']),
        'TEST_RMSE_s_std': 0.0,
        'TEST_MEDAE_s_mean': float(temp_90['test_medae_s']),
        'TEST_MEDAE_s_std': 0.0,
        'TEST_WAPE_mean': float(temp_90['test_wape']),
        'TEST_WAPE_std': 0.0,
        'TEST_RMSLE_mean': float(temp_90['test_rmsle']),
        'TEST_RMSLE_std': 0.0,
        'TEST_R2_mean': float(temp_90['test_r2']),
        'TEST_R2_std': 0.0
    }
]

base_df = pd.DataFrame(baseline_rows)

# Filter out existing baselines if any
clean_ablation = ablation_df[~ablation_df['F'].isin(['Baseline'])]
full_ablation = pd.concat([base_df, clean_ablation], ignore_index=True)

# Save updated test_ablation_summary.csv
full_ablation.to_csv(REFITS_DIR / "test_ablation_summary.csv", index=False)

# Extract Table VI: Baselines + F3 Full Feature models
f3_models = clean_ablation[clean_ablation['F'] == 'F3'].copy()
table_vi = pd.concat([base_df, f3_models], ignore_index=True)
table_vi.to_csv(REFITS_DIR / "table_vi_complete_benchmark.csv", index=False)

print("=== TABLE VI (FINAL BENCHMARK COMPARISON ON 90k) ===")
disp_cols = ['F', 'model', 'TEST_MAE_s_mean', 'TEST_MAE_s_std', 'TEST_RMSE_s_mean', 'TEST_MEDAE_s_mean', 'TEST_WAPE_mean', 'TEST_R2_mean']
print(table_vi[disp_cols].to_string(index=False))
