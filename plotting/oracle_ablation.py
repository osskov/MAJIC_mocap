"""
Plots RMSE (MAG axis) across the acc/mag oracle combos, from the statistics file
produced by experiments/oracle_ablation.py — one distribution plot + heatmap per
base method (mag_on, mag_off, mag_adapt, ekf), comparing its real/perfect acc x
real/perfect mag combos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import numpy as np

import paths
from plotting.utils import plot_metric_distribution, plot_metric_heatmap
from experiments.experiment_utils import load_statistics

PLOTS_DIR = paths.plots_dir("oracle_ablation")

COMBO_LABELS = {
    ('real', 'real'): 'Real Acc / Real Mag',
    ('perfect', 'real'): 'Perfect Acc / Real Mag',
    ('real', 'perfect'): 'Real Acc / Perfect Mag',
    ('perfect', 'perfect'): 'Perfect Acc / Perfect Mag',
}


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE across acc/mag oracle combos, per base method.")
    parser.add_argument("--metric", default='median_deg', help="One of the *_deg / *_rad columns from compute_error_stats.")
    parser.add_argument("--show", action="store_true", help="Also display the plots interactively.")
    args = parser.parse_args()

    df = load_statistics("oracle_ablation")
    if df is None:
        print(f"Error: {paths.statistics_path('oracle_ablation')} not found. "
              "Run experiments/oracle_ablation.py first.")
        return

    for col in [c for c in df.columns if c.endswith('_rad')]:
        df[col.replace('_rad', '_deg')] = np.degrees(df[col])

    df = df[(df['axis'] == 'MAG') & df['base_method'].notna()]

    for base_method in sorted(df['base_method'].unique()):
        base_df = df[df['base_method'] == base_method]
        combos = base_df[['method', 'acc_source', 'mag_source']].drop_duplicates()
        order = combos.sort_values(['acc_source', 'mag_source'])['method'].tolist()
        labels = {
            row.method: COMBO_LABELS[(row.acc_source, row.mag_source)]
            for row in combos.itertuples()
        }
        base_plots_dir = PLOTS_DIR / base_method

        plot_metric_distribution(
            base_df, args.metric, group_col='method', group_order=order, plots_dir=base_plots_dir,
            labels=labels, save=True, show=args.show
        )
        plot_metric_heatmap(
            base_df, args.metric, group_col='method', group_order=order, plots_dir=base_plots_dir,
            labels=labels, save=True, show=args.show
        )

    print("\n--- All oracle ablation plots complete ---")


if __name__ == '__main__':
    main()
