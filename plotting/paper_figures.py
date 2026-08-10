"""
Generates paper figures from results/statistics/all_subject_statistics.parquet,
produced by experiments/benchmark_experiment.py (flat columns, one row per
trial_type/method/joint_name/subject/axis, with *_rad error metrics).
# Note: the 'marker' method is the mocap ground truth used to compute these
# error/correlation stats against, so it never appears as a row value — only the
# IMU-derived methods (ekf, mag_off, mag_on, mag_adapt_15, mag_adapt_100, mag_adapt_200) do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Optional

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import load_statistics
from plotting.utils import plot_metric_distribution, plot_metric_heatmap, DEFAULT_JOINT_ORDER

# ==============================================================================
# CONFIGURATION
# ==============================================================================

STATS_NAME = "all_subject"
PLOTS_DIR = paths.plots_dir()

SHOW_PLOTS = True
SAVE_PLOTS = True

SUBJECTS_TO_PLOT = [f"Subject{i:02d}" for i in range(1, 12)]

# Display labels for the method keys written by experiments/benchmark_experiment.py.
METHOD_LABELS = {
    'ekf': 'EKF',
    'unprojected': 'Unprojected',
    'mag_off': 'Mag Off',
    'mag_off_flat': 'Mag Off (Flat)',
    'mag_off_dyn': 'Mag Off (Dynamic)',
    'mag_off_unnormalized': 'Mag Off (Unnormalized)',
    'mag_off_normalized': 'Mag Off (Normalized)',
    'mag_off_rescaled': 'Mag Off (Rescaled)',
    'mag_on': 'Mag On',
    'mag_adapt_15': 'MAJIC (15)',
    'mag_adapt_100': 'MAJIC (100)',
    'mag_adapt_200': 'MAJIC (200)',
}
METHODS_TO_PLOT = ['ekf', 'mag_off', 'mag_off_flat', 'mag_off_dyn', 'mag_off_unnormalized', 'mag_off_normalized', 'mag_off_rescaled', 'mag_on', 'mag_adapt_15', 'mag_adapt_100', 'mag_adapt_200', 'mag_adapt', 'mag_adapt_dyn']

# Left/right joints are pooled together under a common name; list them
# individually (e.g. 'R_Hip') here instead if that's not desired.
RENAME_JOINTS = {
    'R_Hip': 'Hip', 'L_Hip': 'Hip',
    'R_Knee': 'Knee', 'L_Knee': 'Knee',
    'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle',
}

# One of: mean, std, rmse, mae, mad, min, q25, median, q75, max.
METRIC = 'median'
METRIC_UNITS = 'deg'  # 'deg' or 'rad'

# Which rotation-vector axis to summarize for the error-metric plots.
# 'MAG' (rotation magnitude) is the usual choice.
AXIS_TO_PLOT = 'MAG'

# Facet the distribution plot by joint. Set to None for one pooled plot
# across all joints (the heatmap always breaks down by joint regardless).
FACET_BY = ''

# 'strip' (median/IQR whiskers + points), 'box', or 'bar' (mean + 95% CI).
PLOT_STYLE = 'strip'

# --- Data loading ---


def _load_stats() -> Optional[pd.DataFrame]:
    df = load_statistics(STATS_NAME)
    if df is None:
        print(f"Error: statistics file not found at {paths.statistics_path(STATS_NAME)}. "
              f"Run experiments/benchmark_experiment.py first.")
        return None
    for col in [c for c in df.columns if c.endswith('_rad')]:
        df[col.replace('_rad', '_deg')] = np.degrees(df[col])
    df['joint_name'] = df['joint_name'].replace(RENAME_JOINTS)
    return df[df['method'].isin(METHODS_TO_PLOT) & df['subject'].isin(SUBJECTS_TO_PLOT)]


def main():
    metric_col = f"{METRIC}_{METRIC_UNITS}"

    stats_df = _load_stats()
    if stats_df is None:
        return
    print(stats_df.head())

    error_df = stats_df[stats_df['axis'] == AXIS_TO_PLOT]

    plot_metric_distribution(
        error_df, metric_col, group_col='method', group_order=METHODS_TO_PLOT, plots_dir=PLOTS_DIR,
        labels=METHOD_LABELS, plot_type=PLOT_STYLE, facet_by=(FACET_BY or None), facet_order=DEFAULT_JOINT_ORDER,
        save=SAVE_PLOTS, show=SHOW_PLOTS
    )
    plot_metric_heatmap(
        error_df, metric_col, group_col='method', group_order=METHODS_TO_PLOT, plots_dir=PLOTS_DIR,
        labels=METHOD_LABELS, save=SAVE_PLOTS, show=SHOW_PLOTS
    )

    print("\n--- All plotting complete ---")


if __name__ == "__main__":
    main()
