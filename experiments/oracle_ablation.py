"""
Ablation sweep over the acc/mag "oracle" (ground-truth) input sources, crossed
with the base filter methods. Each combo is a parametrized method name (e.g.
'ekf_perfect_acc_perfect_mag', resolved by experiment_utils.resolve_method_spec),
so this reuses the same generate+stats workers as the vanilla pipeline
(benchmark_experiment.py) — just with a different method list.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import sys
import argparse
from functools import partial
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_utils import (
    SUBJECTS, ACTIVITIES, load_all_joint_angles, compute_error_stats, save_statistics,
    run_tracked_grid, generate_joint_angles_worker, compute_stats_worker,
)

ORACLE_BASE_METHODS = ['mag_on', 'mag_off', 'mag_adapt', 'ekf']
ACC_SOURCES = ['real', 'perfect']
MAG_SOURCES = ['real', 'perfect']


def oracle_method_name(base: str, acc_source: str, mag_source: str) -> str:
    if acc_source == 'real' and mag_source == 'real':
        return base
    name = f"{base}_{acc_source}_acc"
    if mag_source != 'real':
        name += f"_{mag_source}_mag"
    return name


def build_oracle_combos(bases: List[str]) -> List[Tuple[str, str, str, str]]:
    """Returns (method_name, base, acc_source, mag_source) for every oracle combo."""
    combos = []
    for base in bases:
        for acc_source in ACC_SOURCES:
            for mag_source in MAG_SOURCES:
                # mag_off always zeroes the magnetometer before the filter runs, so a
                # mag oracle there would be a no-op duplicating the mag_source='real' case.
                if base == 'mag_off' and mag_source == 'perfect':
                    continue
                combos.append((oracle_method_name(base, acc_source, mag_source), base, acc_source, mag_source))
    return combos


def main():
    parser = argparse.ArgumentParser(
        description="Ablation sweep over acc/mag oracle (ground-truth) input sources."
    )
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS)
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES)
    parser.add_argument("--bases", nargs='+', default=ORACLE_BASE_METHODS,
                         help="Base filter methods to cross with the acc/mag oracle combos.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    combos = build_oracle_combos(args.bases)
    oracle_methods = [name for name, _, _, _ in combos]
    methods = ['marker'] + oracle_methods
    print(f"Running oracle ablation with methods: {methods}")

    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]

    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load'] + methods, generate_joint_angles_worker,
                      args.workers, title="ORACLE ABLATION GENERATION")
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['stats'],
                      partial(compute_stats_worker, methods=methods, stats_name="oracle_ablation"),
                      args.workers, title="ORACLE ABLATION STATISTICS")

    all_data_df = load_all_joint_angles(args.subjects, args.activities, methods)
    if all_data_df.empty:
        print("Error: no data was loaded for any subject.")
        return

    summary_stats_df = compute_error_stats(all_data_df)
    if summary_stats_df.empty:
        print("Warning: summary statistics are empty.")
        return

    method_meta = {name: (base, acc_source, mag_source) for name, base, acc_source, mag_source in combos}
    summary_stats_df['base_method'] = summary_stats_df['method'].map(lambda m: method_meta.get(m, (None,) * 3)[0])
    summary_stats_df['acc_source'] = summary_stats_df['method'].map(lambda m: method_meta.get(m, (None,) * 3)[1])
    summary_stats_df['mag_source'] = summary_stats_df['method'].map(lambda m: method_meta.get(m, (None,) * 3)[2])

    path = save_statistics(summary_stats_df, "oracle_ablation")
    print(f"\nSaved oracle ablation statistics to {path}")


if __name__ == '__main__':
    main()
