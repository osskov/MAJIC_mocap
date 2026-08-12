"""
Sensitivity study for the mag_adapt observability threshold.

Two things are measured, and the figure needs both:

  ACCURACY.  Each threshold is just a parametrized method name (mag_adapt_th<value>,
  resolved by experiment_utils.resolve_method_spec), so the sweep reuses the exact same
  generate+stats workers as the vanilla pipeline (benchmark_experiment.py) — only the
  method list differs. Nothing about the filter is re-implemented here.

  DUTY CYCLE.  The threshold's units — (m/s^2)(m/s^3) — mean nothing to a reader, so the
  sweep also records, per trial and joint, the fraction of samples each threshold gates
  and the o^J distribution behind it. That table is what turns the x-axis into "how often
  is the magnetometer switched off", and it is what makes the duty-cycle claim in
  experiment_utils.DEFAULT_MAG_ADAPT_THRESHOLD's comment checkable rather than quoted.
  Computed with _calculate_observability_metric_ on joint-center-projected traces, i.e.
  the identical quantity _run_relative_filter gates on — an unsmoothed, post-projection
  o^J. (sensor_distributions writes a superficially similar per-joint o^J table, but it
  is winsorized and smoothed for plotting, so a duty cycle read off it would not be the
  filter's.)

THE TWO LIMITS ARE PART OF THE SWEEP. mag_adapt gates on `o^J > threshold`, so a
threshold below every sample gates the magnetometer always (mag_off) and one above every
sample never gates it (mag_on). Those two arms are therefore run alongside the swept
thresholds instead of being read out of the benchmark's statistics file: the curve has to
be shown converging to them, and a reference computed by a different invocation on a
different day is not evidence of convergence.

Outputs
-------
    results/statistics/observability_threshold_statistics.parquet   accuracy, one row per
        subject x activity x joint x method x axis, with a 'threshold' column filled in
        for the swept arms and NaN for mag_on/mag_off
    results/experiments/threshold_sensitivity/Subject<NN>/<activity>/gating.parquet
        fraction of samples gated, per joint x threshold
    results/experiments/threshold_sensitivity/Subject<NN>/<activity>/observability.parquet
        the o^J percentile grid per joint, so the duty-cycle panel can be drawn at any
        threshold, not just the swept ones
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import (
    JOINTS, SUBJECTS, ACTIVITIES, DEFAULT_MAG_ADAPT_THRESHOLD,
    load_all_joint_angles, compute_error_stats, save_statistics, load_raw_data,
    run_tracked_grid, generate_joint_angles_worker, compute_stats_worker,
    _calculate_observability_metric_, project_pair_to_joint_center, pipeline_constants,
)
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_DIR = paths.experiment_dir("threshold_sensitivity")
STATS_NAME = "observability_threshold"

# Third-of-a-decade steps, anchored ON the default rather than spanning a round interval, so
# the shipped value is one of the points measured and not something read off between two of
# them.
#
# The span is set by where the curve stops moving, not by round numbers, and it is
# DELIBERATELY LOPSIDED: four thirds of a decade above the default is enough to reach mag_on
# exactly (every arm from 2154 up reproduces it to the last decimal — o^J never exceeds those
# thresholds, so nothing is gated), but the low end needs seven thirds. Gating turns out to
# be very asymmetric: at 46, which still gates 86% of samples, the error is already at
# mag_on, so the sweep has to run down to ~99% gated before the curve starts back toward
# mag_off at all. A grid that stopped at 46 would show a flat curve with an unreached mag_off
# line floating above it, which reads as a discrepancy rather than as the result it is.
#
# These are units of the CORRECTED o^J (see segment_observability). The pre-fix metric's
# difference term was ~100x too small, so the 10-200 range used against it is not
# comparable and would gate essentially everything here.
#
# Rounded to the two decimals threshold_method encodes, so the value the filter is handed
# (parsed back out of the method name by resolve_method_spec) and the value the duty-cycle
# table is computed at are the same number, and the two tables join exactly.
THRESHOLDS = np.round(DEFAULT_MAG_ADAPT_THRESHOLD * 10.0 ** (np.arange(-7, 5) / 3.0), 2)

# The two limits of the sweep (threshold -> 0 and threshold -> inf), run in the same
# invocation as the swept arms. See the module docstring.
REFERENCE_METHODS = ['mag_off', 'mag_on']

# ...and written under this sweep's OWN variant rather than into the shared default
# namespace. results/joint_angles/ is keyed by method name, so a plain 'mag_on.parquet' is
# whatever ran last: a `benchmark_experiment.py` invocation twice overwrote this sweep's
# limits mid-flight, once after the filter's init covariance changed and once after acc_std
# and mag_std were retuned, each time leaving the curve and its own limits as two different
# estimators. A variant costs one subdirectory and makes that structurally impossible.
#
# The swept arms themselves stay in the default namespace: their method names already encode
# every parameter that distinguishes them (see paths.joint_angles_path), and only the limits
# collide with what other experiments write.
REFERENCE_VARIANT = 'threshold_sweep'

# Percentile grid stored for each joint's o^J. Dense in the tails because the duty cycle
# at a useful threshold is a few percent, and a 1%-resolution grid would quantize it.
OBS_PERCENTILES = np.unique(np.concatenate([
    np.linspace(0.0, 100.0, 201),
    np.array([0.1, 0.2, 0.5, 1.0, 2.0, 98.0, 99.0, 99.5, 99.8, 99.9]),
]))


def threshold_method(threshold: float) -> str:
    """The method name for a swept threshold. The .2f is load-bearing: it is what the
    'threshold' column is parsed back out of, and what TestMethodSpec pins."""
    return f"mag_adapt_th{threshold:.2f}"


def threshold_from_method(method: pd.Series) -> pd.Series:
    """Inverse of threshold_method, vectorized. NaN for any non-swept method."""
    return method.str.extract(r'^mag_adapt_th([\d.]+)$')[0].astype(float)

# ==============================================================================
# Duty cycle
# ==============================================================================

def trial_table_path(subject: str, activity: str, table: str) -> Path:
    return EXPERIMENT_DIR / f"Subject{subject}" / activity / f"{table}.parquet"


def joint_observability(plates: Dict[str, PlateTrial]) -> Dict[str, np.ndarray]:
    """o^J per joint for one trial, exactly as the filter computes it: both plates
    projected to the shared joint center, then the per-sample minimum of the two segment
    observabilities. Joints whose sensors are missing from the trial are skipped, matching
    _joint_angles_from_filter."""
    observability = {}
    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            continue
        parent_proj, child_proj = project_pair_to_joint_center(plates[parent], plates[child])
        observability[joint_name] = _calculate_observability_metric_(parent_proj, child_proj)
    return observability


def gating_tables(subject: str, activity: str, plates: Dict[str, PlateTrial],
                  thresholds: np.ndarray) -> Dict[str, pd.DataFrame]:
    """(gating, observability) tables for one trial.

    `gating` is the duty cycle the figure's x-axis is annotated with: the fraction of
    samples on which mag_adapt would zero the magnetometer. The comparison is `>`, not
    `>=`, because that is the comparison _run_relative_filter makes — at a threshold of 0
    the sample-0 pad (o^J = 0) is the one sample that stays ungated, and the table should
    say so rather than round it away.
    """
    observability = joint_observability(plates)
    if not observability:
        return {}

    gating_rows, percentile_rows = [], []
    for joint_name, obs in observability.items():
        for threshold in thresholds:
            gating_rows.append({
                'subject': f"Subject{subject}", 'activity': activity, 'joint': joint_name,
                'threshold': float(threshold),
                'fraction_gated': float(np.mean(obs > threshold)),
                'n_samples': int(obs.size),
            })
        percentile_rows.append(pd.DataFrame({
            'subject': f"Subject{subject}", 'activity': activity, 'joint': joint_name,
            'percentile': OBS_PERCENTILES,
            'observability': np.percentile(obs, OBS_PERCENTILES),
        }))

    return {'gating': pd.DataFrame(gating_rows),
            'observability': pd.concat(percentile_rows, ignore_index=True)}


def gating_worker(row_key, stage_labels: List[str], shared_state: Dict,
                  thresholds: np.ndarray) -> None:
    """Single-stage worker: reload the trial's raw data and write its duty-cycle tables.

    Reloads rather than sharing with the generation pass because loading a trial costs
    ~2 s against the ~20 min of filter runs beside it, and keeping this separate means the
    cheap half can be re-run on its own (--gating-only) when only the threshold grid
    changed."""
    subject, activity = row_key
    stage = stage_labels[0]

    t_start = time.time()
    shared_state[(row_key, stage)] = "Running"
    try:
        tables = gating_tables(subject, activity, load_raw_data(subject, activity), thresholds)
        if not tables:
            shared_state[(row_key, stage)] = "Skipped"
            return None
        for name, df in tables.items():
            path = paths.ensure_parent(trial_table_path(subject, activity, name))
            df.to_parquet(path, engine='pyarrow', index=False)
            paths.write_manifest(path, constants=pipeline_constants(),
                                 subject=f"Subject{subject}", activity=activity,
                                 thresholds=[float(t) for t in thresholds], n_rows=len(df))
        shared_state[(row_key, f"{stage}_time")] = time.time() - t_start
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None


# ==============================================================================
# Cross-arm consistency
# ==============================================================================
# results/joint_angles/ is a SHARED namespace keyed by method name, so any script that
# runs 'mag_on' overwrites this sweep's copy of it. That is normally harmless — it is the
# same computation — but it stops being harmless the moment the filter or its constants
# change between the two runs, and then the failure is silent: the curve and the reference
# lines are different estimators, every arm still loads, and the only symptom is a number
# that should be impossible.
#
# It has now happened twice on this experiment. A mid-run `benchmark_experiment.py`
# rewrote mag_on/mag_off with a retuned acc_std/mag_std, which put the reference lines
# ~6 deg away from a swept arm that gates 0.2% of samples and therefore IS mag_on.
#
# Every artifact already carries the constants it was written with, in its manifest
# sidecar. So this reads them back and refuses to let the figure treat a mixed set as one
# experiment. Provenance that is written but never checked only tells you what went wrong
# afterwards; checked, it stops the plot.


def arm_constants(subject: str, activity: str, method: str,
                  variant: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The constants recorded in one arm's manifest sidecar, or None if it is missing."""
    path = paths.manifest_path(paths.joint_angles_path(subject, activity, method, variant=variant))
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle).get('constants')


def sweep_arms(thresholds: np.ndarray, with_references: bool = True) -> List[Tuple[str, Optional[str]]]:
    """Every arm the figure reads, as (method, variant) pairs. One list, so the generator,
    the aggregator and the consistency guard cannot disagree about where an arm lives."""
    arms: List[Tuple[str, Optional[str]]] = [(threshold_method(t), None) for t in thresholds]
    if with_references:
        arms += [(method, REFERENCE_VARIANT) for method in REFERENCE_METHODS]
    return arms


def constants_disagreements(arms: List[Tuple[str, Optional[str]]],
                            subjects: Optional[List[str]] = None,
                            activities: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Maps each distinct set of pipeline constants found on disk to the arms carrying it.

    A healthy sweep returns exactly one entry. More than one means the arms were produced
    by different filter configurations and cannot be compared, whatever the file names say.
    The `mag_adapt_threshold` entry is dropped before comparing: it is the swept parameter,
    so it legitimately differs per arm and every arm would look unique.
    """
    groups: Dict[str, List[str]] = {}
    for subject in (subjects or SUBJECTS):
        for activity in (activities or ACTIVITIES):
            for method, variant in arms:
                constants = arm_constants(subject, activity, method, variant)
                if constants is None:
                    continue
                comparable = {k: v for k, v in constants.items() if k != 'mag_adapt_threshold'}
                key = json.dumps(comparable, sort_keys=True)
                label = f"Subject{subject}/{activity}/{method}"
                groups.setdefault(key, []).append(f"{label} [{variant}]" if variant else label)
    return groups


def sweep_stds(thresholds: np.ndarray, subjects: Optional[List[str]] = None,
               activities: Optional[List[str]] = None) -> Dict[str, float]:
    """The gyro/acc/mag stds the SWEPT arms already on disk were written with.

    Read back rather than taken from the module constants, so regenerating the reference
    arms reproduces the sweep's tuning by construction instead of by someone retyping three
    numbers correctly. Raises if the swept arms are themselves mixed — there is then no
    single tuning to match and only a full re-run will do."""
    groups = constants_disagreements(sweep_arms(thresholds, with_references=False),
                                     subjects, activities)
    if not groups:
        raise ValueError("No swept arms on disk to read a tuning from; run the sweep first.")
    if len(groups) > 1:
        raise ValueError("The swept arms are themselves a mix of tunings, so there is no "
                         "single one for the references to match.\n" + describe_disagreements(groups))
    constants = json.loads(next(iter(groups)))
    return {key: constants[key] for key in ('gyro_std', 'acc_std', 'mag_std') if key in constants}


def describe_disagreements(groups: Dict[str, List[str]]) -> str:
    """A message naming what differs and which arms sit on each side."""
    lines = [f"Arms on disk were written with {len(groups)} DIFFERENT filter configurations; "
             f"they are not comparable:"]
    for key, arms in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        constants = json.loads(key)
        shown = {k: v for k, v in constants.items() if k != 'expected_gravity'}
        lines.append(f"  {len(arms):4d} arms  {shown}")
        lines.append(f"           e.g. {', '.join(sorted(arms)[:3])}")
    lines.append("Re-run the sweep so every arm comes from one invocation: "
                 "python -m experiments.threshold_sensitivity")
    return "\n".join(lines)


def load_trial_table(table: str, subjects: Optional[List[str]] = None,
                     activities: Optional[List[str]] = None) -> pd.DataFrame:
    """Concatenates one of the per-trial tables across every trial on disk. Missing trials
    are skipped silently — the sweep is routinely run on a subset."""
    frames = []
    for subject in (subjects or SUBJECTS):
        for activity in (activities or ACTIVITIES):
            path = trial_table_path(subject, activity, table)
            if path.exists():
                frames.append(pd.read_parquet(path, engine='pyarrow'))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS)
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES)
    parser.add_argument("--thresholds", nargs='+', type=float, default=list(THRESHOLDS))
    parser.add_argument("--skip-references", action="store_true",
                        help="Omit the mag_on/mag_off limit arms (they are ~20%% of the runtime).")
    parser.add_argument("--gating-only", action="store_true",
                        help="Write only the duty-cycle tables; skip the filter runs entirely.")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip generation and re-aggregate the joint angles already on disk.")
    parser.add_argument("--references-only", action="store_true",
                        help="Regenerate just the mag_on/mag_off limits, pinned to the tuning "
                             "the swept arms on disk already carry (see sweep_stds), then "
                             "re-aggregate. Repairs a sweep whose limits were overwritten by "
                             "another experiment without redoing the swept arms.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Rounded here too, not just in THRESHOLDS, so a --thresholds override keeps the
    # method-name/duty-cycle-table correspondence rather than quietly losing it.
    thresholds = np.round(np.array(sorted(args.thresholds), dtype=float), 2)
    references = [] if args.skip_references else REFERENCE_METHODS
    swept = [threshold_method(t) for t in thresholds]
    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]

    # In a full run the references inherit the module's current tuning, same as the swept
    # arms beside them. In a --references-only repair they are pinned to whatever the swept
    # arms already on disk were written with, which is the whole point of that mode.
    reference_stds = None
    if args.references_only:
        try:
            reference_stds = sweep_stds(thresholds, args.subjects, args.activities)
        except ValueError as e:
            print(f"Error: {e}")
            return
        print(f"Pinning the limit arms to the swept arms' own tuning: {reference_stds}")

    # 1. Duty cycle. Cheap and independent of the filter runs, so it goes first and is the
    #    only thing --gating-only does. Skipped in a references-only repair, which changes
    #    nothing about o^J.
    if not args.references_only:
        run_tracked_grid(row_keys, ['Subject', 'Activity'], ['gating'],
                         partial(gating_worker, thresholds=thresholds),
                         args.workers, title="OBSERVABILITY DUTY CYCLE")
        if args.gating_only:
            print(f"\nWrote duty-cycle tables under {EXPERIMENT_DIR}")
            return

    # 2. Accuracy. The swept arms sit in the default namespace; the two limits get this
    #    sweep's own variant so nothing else can overwrite them (see REFERENCE_VARIANT).
    if not (args.stats_only or args.references_only):
        print(f"Running threshold sweep with methods: {['marker'] + swept}")
        run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load', 'marker'] + swept,
                         generate_joint_angles_worker, args.workers,
                         title="OBSERVABILITY THRESHOLD GENERATION")
    if references and not args.stats_only:
        print(f"Generating limit arms {references} under variant '{REFERENCE_VARIANT}'")
        run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load'] + references,
                         partial(generate_joint_angles_worker, variant=REFERENCE_VARIANT,
                                 stds=reference_stds),
                         args.workers, title="OBSERVABILITY THRESHOLD LIMIT ARMS")

    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['stats'],
                     partial(compute_stats_worker, methods=['marker'] + swept,
                             stats_name=STATS_NAME),
                     args.workers, title="OBSERVABILITY THRESHOLD STATISTICS")

    # Loaded in two passes because the arms live in two namespaces. 'marker' is the ground
    # truth and runs no filter, so one copy of it serves both.
    frames = [load_all_joint_angles(args.subjects, args.activities, ['marker'] + swept)]
    if references:
        frames.append(load_all_joint_angles(args.subjects, args.activities, references,
                                            variant=REFERENCE_VARIANT))
    all_data_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()
    if all_data_df.empty:
        print("Error: no data was loaded for any subject.")
        return

    groups = constants_disagreements(sweep_arms(thresholds, with_references=bool(references)),
                                     args.subjects, args.activities)
    if len(groups) > 1:
        print("\n" + describe_disagreements(groups))
        print("\nAggregating anyway so the tables exist, but the figure will refuse to draw.")

    summary_stats_df = compute_error_stats(all_data_df)
    if summary_stats_df.empty:
        print("Warning: summary statistics are empty.")
        return

    # The swept threshold is recoverable from the method name; mag_on/mag_off keep NaN, so
    # a plot can select the sweep with a single notna() and still see the two limits.
    summary_stats_df['threshold'] = threshold_from_method(summary_stats_df['method'])

    path = save_statistics(summary_stats_df, STATS_NAME,
                           thresholds=[float(t) for t in thresholds],
                           reference_methods=references,
                           reference_variant=REFERENCE_VARIANT if references else None,
                           reference_stds=reference_stds)
    print(f"\nSaved observability threshold statistics to {path}")
    print(f"Duty-cycle tables under {EXPERIMENT_DIR}")


if __name__ == '__main__':
    main()
