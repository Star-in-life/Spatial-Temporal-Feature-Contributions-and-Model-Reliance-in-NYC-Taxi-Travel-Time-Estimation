#!/usr/bin/env python3
"""
run_1m_2015/code/plot_scaling_publication.py
Generates camera-ready, publication-quality figures for the FJCAI 2027 paper manuscript and defense slides:
1. Full Scaling Benchmark (Figure 1 in Paper extension): All models + baselines (OSRM, TEMP, LightGBM, CatBoost, RealMLP, TabM)
2. Competitive Learner Zoom (Figure 1b): 190s - 215s region highlighting RealMLP crossover and tree saturation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

WORKSPACE = Path("d:/Download/DAP391m/run_1m_2015").resolve()
RES_DIR = WORKSPACE / "results/scaling"
DATA_FILE = RES_DIR / "scaling_summary_aggregated.csv"

df = pd.read_csv(DATA_FILE)

# Color and style definitions conforming to academic publication standards
COLORS = {
    'RealMLP': '#D9381E',    # Deep Crimson Red (Hero)
    'CatBoost': '#2E7D32',   # Forest Green
    'LightGBM': '#1976D2',   # Cobalt Blue
    'TabM': '#E65100',       # Burnt Orange
    'TEMP': '#6A1B9A',       # Deep Purple
    'OSRM': '#757575'        # Slate Grey (Baseline)
}

MARKERS = {
    'RealMLP': 'o',
    'CatBoost': 's',
    'LightGBM': '^',
    'TabM': 'D',
    'TEMP': 'v',
    'OSRM': 'x'
}

LINESTYLES = {
    'RealMLP': '-',
    'CatBoost': '-',
    'LightGBM': '--',
    'TabM': '-.',
    'TEMP': ':',
    'OSRM': '--'
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# =========================================================================
# FIGURE 1: Full Range Scaling Law (Log-Scale Data Size)
# =========================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

models_order = ['OSRM', 'TEMP', 'LightGBM', 'CatBoost', 'TabM', 'RealMLP']

for m in models_order:
    m_df = df[df['model'] == m].sort_values('train_size')
    x = m_df['train_size'].to_numpy() / 1000.0  # in thousands (k)
    y = m_df['test_mae_mean'].to_numpy()
    y_std = m_df['test_mae_std'].to_numpy()
    
    lw = 2.2 if m == 'RealMLP' else 1.6
    ms = 6 if m == 'RealMLP' else 5
    
    ax.plot(x, y, label=m, color=COLORS[m], marker=MARKERS[m], 
            linestyle=LINESTYLES[m], linewidth=lw, markersize=ms, zorder=4 if m == 'RealMLP' else 3)
    
    if (y_std > 0).any():
        ax.fill_between(x, y - y_std, y + y_std, color=COLORS[m], alpha=0.15, zorder=2)

ax.set_xscale('log')
ax.set_xticks([90, 180, 360, 540, 720, 900, 1080])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Training Set Size (k trips, Log Scale)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel("Out-of-Distribution TEST MAE (seconds)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_title("Data Scaling Trajectories: Neural vs. Tree vs. Baselines (2015 NYC Taxi TLC)", 
             fontsize=12, fontweight='bold', pad=12)

ax.legend(frameon=True, facecolor='white', framealpha=0.92, loc='center right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

fig.tight_layout()
p_png = RES_DIR / "figure_scaling_law_all_models.png"
p_pdf = RES_DIR / "figure_scaling_law_all_models.pdf"
fig.savefig(p_png, dpi=300)
fig.savefig(p_pdf)
plt.close(fig)
print(f"Saved: {p_png}")

# =========================================================================
# FIGURE 2: Competitive Learners Zoom (190s - 215s)
# =========================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

comp_models = ['LightGBM', 'CatBoost', 'TabM', 'RealMLP']

for m in comp_models:
    m_df = df[df['model'] == m].sort_values('train_size')
    x = m_df['train_size'].to_numpy() / 1000.0
    y = m_df['test_mae_mean'].to_numpy()
    y_std = m_df['test_mae_std'].to_numpy()
    
    lw = 2.4 if m == 'RealMLP' else 1.8
    ms = 7 if m == 'RealMLP' else 5.5
    
    ax.plot(x, y, label=f"{m} (3-seed Mean ± SD)", color=COLORS[m], marker=MARKERS[m], 
            linestyle=LINESTYLES[m], linewidth=lw, markersize=ms, zorder=4 if m == 'RealMLP' else 3)
    
    ax.fill_between(x, y - y_std, y + y_std, color=COLORS[m], alpha=0.18, zorder=2)

ax.set_xscale('log')
ax.set_xticks([90, 180, 360, 540, 720, 900, 1080])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylim(186.5, 216.5)
ax.set_xlabel("Training Set Size (k trips, Log Scale)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel("TEST MAE (seconds)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_title("Competitive Learners Scaling: Neural Breakout and Tree Saturation (Nov-Dec 2015 TEST)", 
             fontsize=12, fontweight='bold', pad=12)

# Annotate breakout region (clean bottom-left placement below curves)
ax.annotate('RealMLP Breakout (360k)\nMAE = 195.14 ± 0.24s', 
            xy=(360, 195.14), xytext=(115, 189.2),
            arrowprops=dict(facecolor='#D9381E', edgecolor='#D9381E', shrink=0.08, width=1.1, headwidth=5.5),
            fontsize=9.2, fontweight='bold', color='#D9381E',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="#D9381E", lw=1))

# Annotate optimal zone (clean bottom-center placement below RealMLP curve, zero overlap with legend or tree lines)
ax.annotate('Optimal Scaling Zone (720k)\nRealMLP MAE = 192.72 ± 0.70s', 
            xy=(720, 192.72), xytext=(390, 188.0),
            arrowprops=dict(facecolor='#D9381E', edgecolor='#D9381E', shrink=0.08, width=1.1, headwidth=5.5),
            fontsize=9.2, fontweight='bold', color='#D9381E',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="#D9381E", lw=1))

ax.legend(frameon=True, facecolor='white', framealpha=0.92, loc='upper left', fontsize=9.2)
ax.grid(True, linestyle='--', alpha=0.5)

fig.tight_layout()
p_zoom_png = RES_DIR / "figure_scaling_law_competitive_zoom.png"
p_zoom_pdf = RES_DIR / "figure_scaling_law_competitive_zoom.pdf"
fig.savefig(p_zoom_png, dpi=300)
fig.savefig(p_zoom_pdf)
plt.close(fig)
print(f"Saved: {p_zoom_png}")
