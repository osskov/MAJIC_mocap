"""
Plots RMSE (MAG axis) vs. each swept RelativeFilter noise parameter (gyro/acc/mag
std), from the statistics file produced by experiments/noise_sensitivity.py. Each
panel marginalizes over the other two swept parameters (mean RMSE at each level of
the parameter being plotted) — the combos are a full grid product, not a marginal
sweep, so this is the simplest way to see each parameter's effect on its own.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import paths
from plotting import utils  # noqa: F401 (applies shared rcParams styling on import)
from experiments.experiment_utils import load_statistics

PLOTS_DIR = paths.plots_dir()
STD_PARAMS = ['gyro_std', 'acc_std', 'mag_std']


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE vs. each swept noise parameter.")
    parser.add_argument("--method", choices=['mag_on', 'mag_off'], default='mag_on')
    parser.add_argument("--run-tag", default='all', help="Matches the run_tag used by experiments/noise_sensitivity.py "
                                                            "(a subject ID for single-subject runs, or 'all').")
    parser.add_argument("--show", action="store_true", help="Also display the plot interactively.")
    args = parser.parse_args()

    name = f"noise_sensitivity_{args.method}_{args.run_tag}"
    df = load_statistics(name)
    if df is None:
        print(f"Error: {paths.statistics_path(name)} not found. "
              f"Run experiments/noise_sensitivity.py --method {args.method} first.")
        return

    df = df[df['axis'] == 'MAG'].copy()
    df['rmse_deg'] = np.degrees(df['rmse_rad'])

    params = [p for p in STD_PARAMS if p in df.columns and df[p].notna().any()]
    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, param in zip(axes, params):
        marginal = df.groupby(param)['rmse_deg'].mean().reset_index()
        sns.lineplot(data=marginal, x=param, y='rmse_deg', marker='o', ax=ax)
        ax.set_xscale('log')
        ax.set_xlabel(param)
        ax.set_ylabel('Mean RMSE MAG (degrees)' if ax is axes[0] else '')

    fig.suptitle(f'Noise Parameter Sensitivity ({args.method}, {args.run_tag})', fontweight='bold')
    fig.tight_layout()

    out_path = paths.ensure_parent(PLOTS_DIR / f"noise_sensitivity_{args.method}_{args.run_tag}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
