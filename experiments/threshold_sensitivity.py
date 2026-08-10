"""
Sensitivity study for the mag_adapt observability threshold. Each threshold is
just a parametrized method name (mag_adapt_th<value>, resolved by
experiment_utils.resolve_method_spec), so this reuses the exact same
generate+stats workers as the vanilla pipeline (benchmark_experiment.py)
— just with a different method list.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import sys
import argparse
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_utils import (
    SUBJECTS, ACTIVITIES, load_all_joint_angles, compute_error_stats, save_statistics,
    run_tracked_grid, generate_joint_angles_worker, compute_stats_worker,
)

THRESHOLDS = np.logspace(np.log10(10), np.log10(200), 8)


def threshold_method(threshold: float) -> str:
    return f"mag_adapt_th{threshold:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Sensitivity study for the mag_adapt observability threshold.")
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS)
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES)
    parser.add_argument("--thresholds", nargs='+', type=float, default=list(THRESHOLDS))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    methods = ['marker'] + [threshold_method(th) for th in args.thresholds]
    print(f"Running threshold sweep with methods: {methods}")

    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]

    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load'] + methods, generate_joint_angles_worker,
                      args.workers, title="OBSERVABILITY THRESHOLD GENERATION")
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['stats'],
                      partial(compute_stats_worker, methods=methods, stats_name="observability_threshold"),
                      args.workers, title="OBSERVABILITY THRESHOLD STATISTICS")

    all_data_df = load_all_joint_angles(args.subjects, args.activities, methods)
    if all_data_df.empty:
        print("Error: no data was loaded for any subject.")
        return

    summary_stats_df = compute_error_stats(all_data_df)
    if summary_stats_df.empty:
        print("Warning: summary statistics are empty.")
        return

    # threshold is recoverable from the method name for every mag_adapt_th* row.
    is_swept = summary_stats_df['method'].str.startswith('mag_adapt_th')
    summary_stats_df.loc[is_swept, 'threshold'] = (
        summary_stats_df.loc[is_swept, 'method'].str.extract(r'mag_adapt_th([\d.]+)')[0].astype(float)
    )

    path = save_statistics(summary_stats_df, "observability_threshold")
    print(f"\nSaved observability threshold statistics to {path}")


if __name__ == '__main__':
    main()
