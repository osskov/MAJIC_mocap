"""
Aggregate plot (same shape as plotting/paper_figures.py) comparing EKF real/real
against its acc/mag oracle variants, plus mag_adapt as a reference point. Reads
results/statistics/ekf_oracle_comparison_statistics.parquet, produced by
experiments/ekf_oracle_comparison.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import numpy as np

import paths
from plotting.utils import plot_metric_distribution, plot_metric_heatmap, DEFAULT_JOINT_ORDER
from experiments.experiment_utils import load_statistics

PLOTS_DIR = paths.plots_dir("ekf_oracle_comparison")

METHOD_ORDER = ['ekf', 'ekf_perfect_mag', 'ekf_perfect_acc', 'mag_adapt']
METHOD_LABELS = {
    'ekf': 'EKF (Real/Real)',
    'ekf_perfect_mag': 'EKF (Real Acc / Oracle Mag)',
    'ekf_perfect_acc': 'EKF (Oracle Acc / Real Mag)',
    'mag_adapt': 'MAJIC (Real/Real)',
}

RENAME_JOINTS = {
    'R_Hip': 'Hip', 'L_Hip': 'Hip',
    'R_Knee': 'Knee', 'L_Knee': 'Knee',
    'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle',
}

METRIC = 'median'
METRIC_UNITS = 'deg'
AXIS_TO_PLOT = 'MAG'
PLOT_STYLE = 'strip'


def main():
    parser = argparse.ArgumentParser(description="Plot EKF oracle comparison.")
    parser.add_argument("--facet-by-joint", action="store_true", help="One panel per joint instead of pooled.")
    parser.add_argument("--show", action="store_true", help="Also display the plots interactively.")
    args = parser.parse_args()

    df = load_statistics("ekf_oracle_comparison")
    if df is None:
        print(f"Error: {paths.statistics_path('ekf_oracle_comparison')} not found. "
              "Run experiments/ekf_oracle_comparison.py first.")
        return

    for col in [c for c in df.columns if c.endswith('_rad')]:
        df[col.replace('_rad', '_deg')] = np.degrees(df[col])
    df['joint_name'] = df['joint_name'].replace(RENAME_JOINTS)
    df = df[df['method'].isin(METHOD_ORDER)]

    metric_col = f"{METRIC}_{METRIC_UNITS}"
    error_df = df[df['axis'] == AXIS_TO_PLOT]

    plot_metric_distribution(
        error_df, metric_col, group_col='method', group_order=METHOD_ORDER, plots_dir=PLOTS_DIR,
        labels=METHOD_LABELS, plot_type=PLOT_STYLE, facet_by=('joint_name' if args.facet_by_joint else None),
        facet_order=DEFAULT_JOINT_ORDER, save=True, show=args.show
    )
    plot_metric_heatmap(
        error_df, metric_col, group_col='method', group_order=METHOD_ORDER, plots_dir=PLOTS_DIR,
        labels=METHOD_LABELS, save=True, show=args.show
    )

    print("\n--- EKF oracle comparison plots complete ---")


if __name__ == '__main__':
    main()
