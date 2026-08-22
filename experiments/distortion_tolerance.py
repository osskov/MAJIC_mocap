"""
How much magnetic distortion the magnetometer-on filter can take before it stops being
worth using.

THE QUESTION. mag_on beats mag_off at the proximal joints and loses to it at the distal
ones (see plotting/paper_figures.py), and the reason is the lab floor's field anomaly,
which grows as a sensor gets closer to it. That is a statement about one lab. What a
reader needs in order to apply the method anywhere else is the DOSE-RESPONSE: at what
level of field distortion does using the magnetometer stop paying for itself? This sweep
answers that by re-running the same mag_on filter with the estimated distortion
multiplied by a scale a in {0, 0.25, 0.5, ..., 2}, and finding where each joint's curve
crosses that joint's own mag_off.

THE DIAL. The distortion is estimated per sensor per sample as the world-frame residual
against the subject's assumed uniform field, and only that residual is scaled — see
experiment_utils._compute_scaled_mag for the definition and for what "distortion" does and
does not include here. The two ends are exact rather than approximate: a = 0 IS the
existing 'perfect_mag' oracle arm and a = 1 IS the ordinary real reading, both bit for
bit. That is what makes the sweep trustworthy — its endpoints are arms the pipeline
already runs, so a curve that failed to land on them at the ends would be a bug and not a
result. Each scale is just a parametrized method name ('mag_on_dist0.50', resolved by
experiment_utils.resolve_method_spec), so the sweep reuses the vanilla generate+stats
workers and no filter code is re-implemented here.

WHY THE SWEEP GOES PAST 100%. The real field is the field this dataset happens to have.
If mag_on still wins at a = 1 for a given joint, a sweep capped at 1 reports "no breaking
point" — which is a property of this lab, not of the method. Scales above 1 amplify the
same spatial distortion pattern, which is what turns the output from "the magnetometer is
fine here" into a tolerance a reader can compare their own lab against.

THE X-AXIS HAS TO BE PHYSICAL. A scale factor means nothing outside this dataset, so the
sweep also records, per scale, how much distortion that actually is, in two forms:

  ABSOLUTE (segment_distortion). The angle between a segment's distorted field and the
  assumed uniform field. This is the heading error the anomaly writes directly into an
  absolute/EKF-style orientation estimate.

  RELATIVE (joint_distortion). The angle between the parent's and the child's distorted
  fields, both in the world frame. This is what a RELATIVE magnetometer correction
  actually suffers from: a distortion common to both sensors cancels in the relative
  update and only the disagreement between them survives. It is zero by construction at
  a = 0 and it is the axis the breakeven should be quoted on.

Both are reported in degrees, which is the same unit as the error axis, so "the
magnetometer stops helping the ankle above N degrees of inter-sensor field disagreement"
is a sentence this experiment can support.

Outputs
-------
    results/statistics/distortion_tolerance_statistics.parquet   accuracy, one row per
        subject x activity x joint x method x axis, with a 'distortion_scale' column filled
        in for the swept arms and NaN for the mag_on/mag_off references
    results/experiments/distortion_tolerance/Subject<NN>/<activity>/segment_distortion.parquet
    results/experiments/distortion_tolerance/Subject<NN>/<activity>/joint_distortion.parquet
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import (
    JOINTS, SUBJECTS, ACTIVITIES, TRIAL_DATASET,
    load_all_joint_angles, compute_error_stats, save_statistics, load_raw_data,
    run_tracked_grid, generate_joint_angles_worker, compute_stats_worker,
    _compute_expected_mag_field, pipeline_constants,
)
from src.toolchest.PlateTrial import PlateTrial

# THIS EXPERIMENT OWNS ITS OWN JOINT-ANGLE TREE:
# results/experiments/distortion_tolerance/joint_angles/. `results/joint_angles/` belongs to
# benchmark_experiment.py alone — see paths.joint_angles_write_path for the failure
# that rule exists to prevent.
EXPERIMENT_NAME = "distortion_tolerance"

EXPERIMENT_DIR = paths.experiment_dir("distortion_tolerance")
STATS_NAME = "distortion_tolerance"

# The requested quarters of the measured distortion, plus two amplified arms (see the
# docstring on why the sweep must not stop at 1.0) and 0.75 to keep the spacing even
# through the region where the distal joints are expected to cross over.
#
# Rounded to the two decimals distortion_method encodes, so the value the filter is handed
# (parsed back out of the method name by resolve_method_spec) and the value the distortion
# tables are computed at are the same number, and the two tables join exactly.
DISTORTION_SCALES = np.round(np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]), 2)

# mag_off is the floor the sweep is measured against — it is the alternative a practitioner
# has when the field is too dirty to use, so the crossing point of the curve with it IS the
# answer to "how much distortion is acceptable". mag_on is a redundancy check rather than a
# limit: 'mag_on_dist1.00' reproduces it exactly by construction, so the two arms landing
# on the same number is evidence the distortion dial is wired up correctly, and one arm
# of runtime is a cheap price for that.
REFERENCE_METHODS = ['mag_off', 'mag_on']

# The scale whose arm must reproduce plain mag_on, used for that check.
UNITY_SCALE = 1.0

# Summary statistics kept for each distortion metric. Distortion is spiky and one-sided
# (the anomaly is a function of where the segment is, and the segment spends most of a trial
# away from the floor), so the median and the 95th percentile say different things and the
# figure needs both: the median is the typical dose, the p95 is the dose during the part of
# the stride that does the damage.
QUANTILES = {'median': 0.5, 'p95': 0.95}


def distortion_method(scale: float) -> str:
    """The method name for a swept distortion scale. The .2f is load-bearing: it is what
    the 'distortion_scale' column is parsed back out of, and what TestMethodSpec pins."""
    return f"mag_on_dist{scale:.2f}"


def scale_from_method(method: pd.Series) -> pd.Series:
    """Inverse of distortion_method, vectorized. NaN for any non-swept method."""
    return method.str.extract(r'^mag_on_dist([\d.]+)$')[0].astype(float)

# ==============================================================================
# How much distortion a scale actually is
# ==============================================================================

def trial_table_path(subject: str, activity: str, table: str) -> Path:
    return EXPERIMENT_DIR / f"Subject{subject}" / activity / f"{table}.parquet"


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-sample angle between two stacks of vectors, in degrees.

    Via arctan2 of the cross and dot products rather than arccos of the normalized dot:
    arccos loses all precision near 0 degrees, which is exactly where the a = 0 arm and the
    small-scale arms live, and would report a few tenths of a degree of numerical noise as
    the distortion at a scale of zero."""
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    dot = np.einsum('...i,...i->...', u, v)
    return np.degrees(np.arctan2(cross, dot))


def _summarize(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """mean + the QUANTILES of one per-sample distortion metric, as flat columns."""
    out = {f'{prefix}_mean': float(np.mean(values))}
    for name, q in QUANTILES.items():
        out[f'{prefix}_{name}'] = float(np.quantile(values, q))
    return out


def scaled_world_fields(plates: Dict[str, PlateTrial], expected_mag: np.ndarray,
                        scale: float) -> Dict[str, np.ndarray]:
    """Each sensor's world-frame magnetic field at one distortion scale.

    The world frame is where the comparison has to happen: two sensors in a uniform field
    read different body-frame vectors purely because they are oriented differently, so a
    body-frame difference would measure their relative orientation and not the field. This
    is the same `e + a * (m_world - e)` the filter is fed (experiment_utils._compute_scaled_mag),
    just left in the world frame instead of rotated back."""
    fields = {}
    for name, plate in plates.items():
        world_mag = np.einsum('nij,nj->ni', plate.world_trace.rotations, plate.imu_trace.mag)
        fields[name] = expected_mag + scale * (world_mag - expected_mag)
    return fields


def distortion_tables(subject: str, activity: str, plates: Dict[str, PlateTrial],
                      scales: np.ndarray) -> Dict[str, pd.DataFrame]:
    """(segment_distortion, joint_distortion) tables for one trial — the physical x-axis
    the accuracy curve is plotted against. See the module docstring for why there are two.

    Both are computed from ground-truth orientations and the raw magnetometer only; no
    filter runs, which is what makes this half of the experiment cheap enough to re-run on
    its own when the scale grid changes (--distortion-only)."""
    expected_mag = _compute_expected_mag_field(list(plates.values()))
    field_norm = float(np.linalg.norm(expected_mag))

    segment_rows, joint_rows = [], []
    for scale in scales:
        fields = scaled_world_fields(plates, expected_mag, float(scale))
        common = {'subject': f"Subject{subject}", 'activity': activity,
                  'distortion_scale': float(scale)}

        for sensor, field in fields.items():
            residual = np.linalg.norm(field - expected_mag, axis=1)
            segment_rows.append({
                **common, 'sensor': sensor,
                # In the arbitrary units the Xsens magnetometer channels are exported in
                # (see global_assumptions.MAG_UNIT), so also given as a fraction of the
                # assumed field's own magnitude, which is unit-free and comparable.
                **_summarize(residual, 'magdev'),
                **_summarize(residual / field_norm, 'magdev_frac'),
                **_summarize(_angle_deg(field, np.broadcast_to(expected_mag, field.shape)),
                             'field_angle_deg'),
                'n_samples': int(len(residual)),
            })

        for joint_name, (parent, child) in JOINTS.items():
            if parent not in fields or child not in fields:
                continue
            disagreement = _angle_deg(fields[parent], fields[child])
            joint_rows.append({
                **common, 'joint': joint_name, 'parent': parent, 'child': child,
                **_summarize(disagreement, 'disagreement_deg'),
                'n_samples': int(len(disagreement)),
            })

    if not segment_rows:
        return {}
    return {'segment_distortion': pd.DataFrame(segment_rows),
            'joint_distortion': pd.DataFrame(joint_rows)}


def distortion_worker(row_key, stage_labels: List[str], shared_state: Dict,
                      scales: np.ndarray) -> None:
    """Single-stage worker: reload the trial's raw data and write its distortion tables.

    Reloads rather than sharing with the generation pass for the same reason
    threshold_sensitivity.gating_worker does — a trial load costs ~3 s against the ~20 min
    of filter runs beside it, and keeping the cheap half separate means it can be re-run
    alone."""
    subject, activity = row_key
    stage = stage_labels[0]

    t_start = time.time()
    shared_state[(row_key, stage)] = "Running"
    try:
        tables = distortion_tables(subject, activity, load_raw_data(subject, activity), scales)
        if not tables:
            shared_state[(row_key, stage)] = "Skipped"
            return None
        for name, df in tables.items():
            path = paths.ensure_parent(trial_table_path(subject, activity, name))
            df.to_parquet(path, engine='pyarrow', index=False)
            paths.write_manifest(path, constants=pipeline_constants(),
                                 subject=f"Subject{subject}", activity=activity,
                                 distortion_scales=[float(s) for s in scales], n_rows=len(df))
        shared_state[(row_key, f"{stage}_time")] = time.time() - t_start
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None


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
# The wiring check
# ==============================================================================

def check_unity_arm(stats: pd.DataFrame, tol_deg: float = 1e-6) -> Optional[str]:
    """Verifies that the a = 1 arm reproduced plain mag_on, returning a complaint or None.

    This is the whole sweep's load-bearing assumption stated as a check on its own output:
    if 'mag_on_dist1.00' and 'mag_on' disagree, the distortion dial is not the identity at
    a = 1, every other scale is therefore measuring something other than a fraction of the
    real distortion, and the curve's position relative to the mag_off floor cannot be
    trusted. Reported rather than raised — the numbers are already on disk by this point
    and a reader is better served by being told they are suspect than by losing them."""
    unity = distortion_method(UNITY_SCALE)
    subset = stats[(stats['axis'] == 'MAG') & (stats['method'].isin([unity, 'mag_on']))]
    wide = subset.pivot_table(index=['subject', 'trial_type', 'joint_name'],
                              columns='method', values='rmse_rad')
    if unity not in wide.columns or 'mag_on' not in wide.columns:
        return None  # one of the two arms was not run; nothing to check against
    gap = np.degrees((wide[unity] - wide['mag_on']).abs()).max()
    if not np.isfinite(gap) or gap > tol_deg:
        return (f"WARNING: {unity} and mag_on differ by up to {gap:.3g} deg, but they are the "
                f"same configuration by construction. The distortion scaling is not the "
                f"identity at a=1 — treat the whole sweep as suspect.")
    return None

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults to ONE subject, unlike the other sweeps: eight arms x seven joints is ~20 min
    # of filter time per trial, and this is a supplementary figure whose shape should be
    # inspected on one subject before it is paid for on eleven.
    parser.add_argument("--subjects", nargs='+', default=['01'],
                        help="Subjects to run (default: 01). Pass every subject explicitly, "
                             "or --all, for the full sweep.")
    parser.add_argument("--all", action="store_true", help="Run every subject.")
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES)
    parser.add_argument("--scales", nargs='+', type=float, default=list(DISTORTION_SCALES),
                        help="Multipliers on the estimated magnetic distortion. 0 is the mag "
                             "oracle, 1 is the real reading, >1 amplifies it.")
    parser.add_argument("--skip-references", action="store_true",
                        help="Omit the mag_off floor and the mag_on check arm.")
    parser.add_argument("--distortion-only", action="store_true",
                        help="Write only the distortion tables; skip the filter runs entirely.")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip generation and re-aggregate the joint angles already on disk.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    subjects = SUBJECTS if args.all else args.subjects
    # Rounded here too, not just in DISTORTION_SCALES, so a --scales override keeps the
    # method-name/distortion-table correspondence rather than quietly losing it.
    scales = np.round(np.array(sorted(args.scales), dtype=float), 2)
    if np.any(scales < 0):
        raise ValueError(f"Distortion scales must be non-negative; got {scales.tolist()}. "
                         f"A negative scale would invert the distortion rather than shrink it.")
    references = [] if args.skip_references else REFERENCE_METHODS
    methods = ['marker'] + references + [distortion_method(s) for s in scales]
    row_keys = [(subject, activity) for subject in subjects for activity in args.activities]

    # 1. How much distortion each scale is. Cheap and independent of the filter runs, so it
    #    goes first and is the only thing --distortion-only does.
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['distortion'],
                     partial(distortion_worker, scales=scales),
                     args.workers, title="MAGNETIC DISTORTION DOSE")
    if args.distortion_only:
        print(f"\nWrote distortion tables under {EXPERIMENT_DIR}")
        return

    # 2. Accuracy. per_cell, unlike the other method-name sweeps: this one is meant to be run
    #    on a single subject, so the default one-process-per-trial shape would put eight arms
    #    behind two processes and leave ten cores idle. Each arm reloads the trial (~3 s
    #    against its own ~3 min), which is the trade run_tracked_grid's per_cell mode exists
    #    for.
    print(f"Running distortion sweep with methods: {methods}")
    if not args.stats_only:
        run_tracked_grid(row_keys, ['Subject', 'Activity'], methods,
                         partial(generate_joint_angles_worker,
                                 experiment=EXPERIMENT_NAME), args.workers,
                         title="DISTORTION TOLERANCE GENERATION", per_cell=True)
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['stats'],
                     partial(compute_stats_worker, methods=methods, stats_name=STATS_NAME,
                             experiment=EXPERIMENT_NAME),
                     args.workers, title="DISTORTION TOLERANCE STATISTICS")

    all_data_df = load_all_joint_angles(TRIAL_DATASET, row_keys, methods,
                                        experiment=EXPERIMENT_NAME)
    if all_data_df.empty:
        print("Error: no data was loaded for any subject.")
        return

    summary_stats_df = compute_error_stats(all_data_df)
    if summary_stats_df.empty:
        print("Warning: summary statistics are empty.")
        return

    # The swept scale is recoverable from the method name; mag_on/mag_off keep NaN, so a plot
    # can select the sweep with a single notna() and still see the references.
    summary_stats_df['distortion_scale'] = scale_from_method(summary_stats_df['method'])

    path = save_statistics(summary_stats_df, STATS_NAME,
                           distortion_scales=[float(s) for s in scales],
                           reference_methods=references)
    print(f"\nSaved distortion tolerance statistics to {path}")
    print(f"Distortion tables under {EXPERIMENT_DIR}")

    complaint = check_unity_arm(summary_stats_df)
    print(complaint if complaint else
          f"Check passed: {distortion_method(UNITY_SCALE)} reproduces mag_on.")


if __name__ == '__main__':
    main()
