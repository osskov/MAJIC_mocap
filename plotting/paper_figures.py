"""
Generates paper figures from results/statistics/all_subject_statistics.parquet,
produced by experiments/benchmark_experiment.py (flat columns, one row per
trial_type/method/joint_name/subject/axis, with *_rad error metrics).
# Note: the 'marker' method is the mocap ground truth used to compute these
# error/correlation stats against, so it never appears as a row value — only the
# IMU-derived methods (ekf, mag_off, mag_on, mag_adapt_15, mag_adapt_100, mag_adapt_200) do.

The figure builders take the statistics name, the output directory and the method
list as arguments so a benchmark-shaped run at a different filter tuning can draw the
identical figures from its own statistics file without a second copy of this module —
see plotting/normalized_benchmark.py. main() supplies the benchmark's own values, so
`python -m plotting.paper_figures` is unchanged.
"""
from pathlib import Path
from typing import Dict, List, Optional

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
    'mag_adapt': 'MAJIC',
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

# The methods compared in Figure 1 — the four the benchmark grid runs
# (experiment_utils.METHODS minus 'marker', which is the ground truth), ordered
# baseline -> ablations -> MAJIC.
FIGURE_1_METHODS = ['ekf', 'mag_off', 'mag_on', 'mag_adapt']

# One of: mean, std, rmse, mae, mad, min, q25, median, q75, max.
METRIC = 'rmse'
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


def _load_stats(stats_name: str = STATS_NAME,
                methods_to_plot: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    df = load_statistics(stats_name)
    if df is None:
        print(f"Error: statistics file not found at {paths.statistics_path(stats_name)}. "
              f"Run the experiment that writes it first.")
        return None
    methods_to_plot = METHODS_TO_PLOT if methods_to_plot is None else methods_to_plot
    for col in [c for c in df.columns if c.endswith('_rad')]:
        df[col.replace('_rad', '_deg')] = np.degrees(df[col])
    df['joint_name'] = df['joint_name'].replace(RENAME_JOINTS)
    return df[df['method'].isin(methods_to_plot) & df['subject'].isin(SUBJECTS_TO_PLOT)]


# --- Figures ---


def figure_1(error_df: pd.DataFrame, metric_col: str, plot_type: str = PLOT_STYLE,
             plots_dir: Path = PLOTS_DIR, group_order: Optional[List[str]] = None,
             labels: Optional[Dict[str, str]] = None, title: str = "Joint Angle Error by Method") -> None:
    """Figure 1: joint-angle error by method — one point per subject/trial/joint,
    summarized per method by median/IQR ('strip') or mean/SD ('bar_sd').

    The style is in the filename so the variants don't overwrite each other."""
    unit_label = 'degrees' if METRIC_UNITS == 'deg' else 'radians'
    suffix = '' if plot_type == 'strip' else f"_{plot_type}"
    plot_metric_distribution(
        error_df, metric_col, group_col='method',
        group_order=FIGURE_1_METHODS if group_order is None else group_order, plots_dir=plots_dir,
        labels=METHOD_LABELS if labels is None else labels,
        plot_type=plot_type, facet_by=(FACET_BY or None), facet_order=DEFAULT_JOINT_ORDER,
        ylabel=f"{METRIC.upper()} ({unit_label})", title=title,
        filename=f"figure_1_{METRIC}_by_method{suffix}.png", save=SAVE_PLOTS, show=SHOW_PLOTS
    )


def make_figures(stats_name: str = STATS_NAME, plots_dir: Path = PLOTS_DIR,
                 figure_methods: Optional[List[str]] = None,
                 methods_to_plot: Optional[List[str]] = None,
                 labels: Optional[Dict[str, str]] = None,
                 title: str = "Joint Angle Error by Method") -> None:
    """The benchmark's two figures — the by-method distribution and the joint x method
    heatmap — drawn from `stats_name` into `plots_dir`.

    Parametrized rather than reading the module constants directly so a re-tuned
    benchmark run (plotting/normalized_benchmark.py) draws the same two figures from its
    own statistics file, into its own directory, under its own method names. The
    filenames are fixed ('figure_1_*.png', 'heatmap_*.png'), so a caller passing a
    different stats_name must pass a different plots_dir or it will overwrite the
    paper's figures."""
    metric_col = f"{METRIC}_{METRIC_UNITS}"
    figure_methods = FIGURE_1_METHODS if figure_methods is None else figure_methods
    methods_to_plot = METHODS_TO_PLOT if methods_to_plot is None else methods_to_plot
    labels = METHOD_LABELS if labels is None else labels

    stats_df = _load_stats(stats_name, methods_to_plot)
    if stats_df is None:
        return
    print(stats_df.head())

    error_df = stats_df[stats_df['axis'] == AXIS_TO_PLOT]

    figure_1(error_df, metric_col, plots_dir=plots_dir, group_order=figure_methods,
             labels=labels, title=title)

    plot_metric_heatmap(
        error_df, metric_col, group_col='method', group_order=methods_to_plot, plots_dir=plots_dir,
        labels=labels, save=SAVE_PLOTS, show=SHOW_PLOTS
    )

    print("\n--- All plotting complete ---")


def main():
    make_figures()


if __name__ == "__main__":
    main()
