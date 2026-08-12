"""
The benchmark's figures, drawn from experiments/normalized_benchmark.py's statistics.

Identical figures to plotting/paper_figures.py — the by-method distribution with its
blocked significance report, and the joint x method heatmap — over the same four arms
under their '_normalized' names, read from results/statistics/normalized_benchmark_statistics.parquet
and written to plots/normalized_benchmark/.

Everything is delegated to paper_figures.make_figures so the two figures cannot drift
apart: what varies here is the statistics name, the output directory and the method
labels, and nothing else. The separate output directory is required, not cosmetic — the
figure filenames are fixed, so drawing into plots/ would overwrite the paper's Figure 1.
"""
import argparse
from pathlib import Path

import paths
from plotting.paper_figures import METHOD_LABELS, make_figures
from experiments.normalized_benchmark import NORMALIZED_METHODS, STATS_NAME

PLOTS_DIR = paths.plots_dir("normalized_benchmark")

# Same display names as the benchmark's, since these are the same four methods — the
# '(Normalized)' qualifier is in the figure title rather than repeated on every tick
# label, where it would just be noise given that every arm carries it.
NORMALIZED_LABELS = {**METHOD_LABELS,
                     **{f"{base}_normalized": METHOD_LABELS.get(base, base)
                        for base in ('ekf', 'mag_off', 'mag_on', 'mag_adapt')}}

TITLE = "Joint Angle Error by Method (Normalized, Re-tuned)"


def main():
    parser = argparse.ArgumentParser(
        description="Draw the benchmark figures from the normalized/re-tuned run's statistics."
    )
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR),
                         help="Output directory for the figures.")
    args = parser.parse_args()

    make_figures(stats_name=STATS_NAME, plots_dir=Path(args.plots_dir),
                 figure_methods=NORMALIZED_METHODS, methods_to_plot=NORMALIZED_METHODS,
                 labels=NORMALIZED_LABELS, title=TITLE)


if __name__ == "__main__":
    main()
