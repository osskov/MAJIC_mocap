"""
Plots RMSE (MAG axis) vs. the mag_adapt observability threshold, from the
statistics file produced by experiments/threshold_sensitivity.py. The threshold is
a continuous swept parameter, so this is a line plot (log x-axis) rather than the
categorical distribution/heatmap comparisons in plotting.utils — it still borrows
that module's styling and PLOTS_DIR convention.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import paths
from plotting import utils  # noqa: F401 (applies shared rcParams styling on import)
from experiments.experiment_utils import load_statistics

PLOTS_DIR = paths.plots_dir()


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE vs. mag_adapt observability threshold.")
    parser.add_argument("--show", action="store_true", help="Also display the plot interactively.")
    args = parser.parse_args()

    df = load_statistics("observability_threshold")
    if df is None:
        print(f"Error: {paths.statistics_path('observability_threshold')} not found. "
              "Run experiments/threshold_sensitivity.py first.")
        return

    df = df[(df['axis'] == 'MAG') & df['threshold'].notna()].copy()
    if df.empty:
        print("No mag_adapt_th* rows found in the statistics file.")
        return
    df['rmse_deg'] = np.degrees(df['rmse_rad'])

    plot_df = df.groupby(['subject', 'threshold'])['rmse_deg'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plot_df, x='threshold', y='rmse_deg', hue='subject', marker='o', alpha=0.5, ax=ax, legend=False)
    sns.lineplot(data=plot_df, x='threshold', y='rmse_deg', color='black', linewidth=3, errorbar=None,
                 label='Mean', marker='s', ax=ax)

    ax.set_xscale('log')
    ax.set_xlabel('Mag Adapt Threshold')
    ax.set_ylabel('RMSE MAG (degrees)')
    ax.set_title('Sensitivity of RMSE MAG to Mag Adapt Threshold')

    out_path = paths.ensure_parent(PLOTS_DIR / "observability_threshold_sensitivity.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
