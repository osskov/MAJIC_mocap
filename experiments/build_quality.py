"""What the build did to the data, measured per processing step across every trial.

A build produces 281 artifacts and a scroll of warnings. Whether any given one is trustworthy
is answerable today only by rebuilding it and watching the terminal, and whether a pipeline
change improved or damaged things is not answerable at all. This turns both into tables.

THREE TIERS, and the split matters because it decides what can disagree with what.

  Tier 1 is the build itself, via building/report.BuildReport. Reconstruction residuals,
  per-plate sync lags, alignment rotations -- all of it already computed and, until now,
  printed and dropped. Free, and it is the only tier that can see inside a build.

  Tier 2 is this module: it reads those sidecars plus the built parquets and derives what
  tier 1 cannot know because it only ever sees one trial -- population spreads, per-sensor
  pooling, dataset comparisons.

  Tier 3 is plotting/build_quality.py, which reads ONLY the tables written here. Nothing in
  the plotting layer reloads a trial, so no number in a figure can disagree with the parquet
  beside it.

WHAT THIS DELIBERATELY DOES NOT DO. It does not gate a build, it does not change what a build
produces, and it runs no filters. It reports. A health score appears, and it is a triage aid
printed beside its own components, never a pass/fail.

REPLICATE STRUCTURE IS MEASURED, NOT ASSUMED. IMoVE's THIGH_L_H, _M and _L share one
WorldTrace and one reconstruction, so every reconstruction-derived metric is bit-identical
across the three and counting them as three replicates inflates n threefold on exactly the
numbers the report leads with. Alignment and lever-arm metrics are per-sensor and do
replicate. `icc_report` measures which is which rather than taking anyone's word, and the
blocking follows from it.

    python -m experiments.build_quality --dataset imove
    python -m experiments.build_quality --dataset alborno --only-tables reconstruction sync
"""
import argparse
import os

os.environ.setdefault("DISABLE_TQDM", "True")

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import (TRIAL_DATASET, build_report_path,
                                          cached_trial_status)
from experiments.experiment_utils import RESIDUAL_WARN_FRACTION
from src.toolchest.building.report import STEPS
from src.toolchest.building.sources import get_source

EXPERIMENT_NAME = "build_quality"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# The spine every pooled table is reported on, matching sensor_distributions so the two pool
# together without anyone re-deciding what a summary is.
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Per metric family, what counts as one independent observation. Derived from measured ICCs
# (see `icc_report`) rather than asserted: the three IMoVE placements on a segment share a
# reconstruction, so blocking reconstruction metrics on the sensor would trebly count one
# measurement, while alignment genuinely differs per sensor.
BLOCKING = {
    # One export file is one observation: packet loss and out-of-range samples are properties
    # of a radio link and an export, not of the segment underneath the sensor.
    'S1_parse': ('dataset', 'subject', 'trial', 'entity'),
    'S2_reconstruction': ('dataset', 'subject', 'trial', 'entity'),
    'S3_pairing': ('dataset', 'subject', 'trial'),
    'S4_sync': ('dataset', 'subject', 'trial'),
    # Blocked on the entity, unlike reconstruction: the discarded-power fraction is a property
    # of what the sensor measured, so a segment's three placements are three real replicates.
    'S5_resampling': ('dataset', 'subject', 'trial', 'entity'),
    'S6_timeline': ('dataset', 'subject', 'trial'),
    'S7_alignment': ('dataset', 'subject', 'trial', 'entity'),
    'S8_lever_arm': ('dataset', 'subject', 'trial', 'entity'),
}

# An ICC at or above this means the grouping's members are not independent replicates.
ICC_DEPENDENT = 0.95

TRIAL_TABLES = ('index', 'report_rows', 'parse', 'reconstruction', 'sync', 'resampling',
                'timeline', 'alignment', 'lever_arm', 'icc_report', 'health', 'coverage',
                'invalid_sections', 'plate_diagnostics')

# A run of invalid frames shorter than this is a blink, not a gap. Reported separately so a
# trial with one dropped frame is not filed beside one missing a whole limb.
SHORT_INVALID_RUN = 5


def analysis_constants(dataset: str) -> Dict[str, object]:
    """Everything a reader needs to reproduce these numbers, recorded in every manifest."""
    return {'dataset': dataset, 'quantiles': QUANTILES, 'blocking': {k: list(v) for k, v
                                                                     in BLOCKING.items()},
            'icc_dependent_threshold': ICC_DEPENDENT, 'steps': list(STEPS)}


def _dataset_dir(dataset: str) -> Path:
    return EXPERIMENT_DIR / dataset


def _save(frame: pd.DataFrame, name: str, dataset: str, **manifest_extra) -> Path:
    path = _dataset_dir(dataset) / f'{name}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(dataset),
                         experiment=EXPERIMENT_NAME, table=name, n_rows=len(frame),
                         **manifest_extra)
    return path


# ---------------------------------------------------------------------------- tier 2 loading

def load_build_reports(dataset: str) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Every trial's BuildReport sidecar, concatenated. Also returns what was unavailable.

    A missing sidecar is NOT staleness -- the report is not part of the cache key, so a trial
    built before the instrumentation existed is perfectly valid and simply has no tier-1 data.
    Reported rather than raised, because on a partially rebuilt tree that is the normal state
    and failing here would make the analysis unusable exactly when it is most wanted.
    """
    rows, unavailable = [], []
    for subject, trial in get_source(dataset).enumerate_trials():
        path = build_report_path(dataset, subject, trial)
        if not path.exists():
            unavailable.append((subject, trial))
            continue
        frame = pd.read_parquet(path)
        frame.insert(0, 'trial', trial)
        frame.insert(0, 'subject', subject)
        frame.insert(0, 'dataset', dataset)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(), unavailable
    return pd.concat(rows, ignore_index=True), unavailable


def build_index(dataset: str) -> pd.DataFrame:
    """One row per enumerated trial: cache status and the manifest's headline scalars.

    This is the triage table. It is the only one that covers trials which FAILED to build --
    and a failure taxonomy is most of the value of looking at 281 trials at once.

    A failed trial has no parquet, but since `build_trials` started writing the sidecar from
    the failure path it may still have a PARTIAL report. `last_step_reported` is filled in by
    `run` from the concatenated reports, and for a failure it names how far the build got.
    """
    rows = []
    for subject, trial in get_source(dataset).enumerate_trials():
        status, reason = cached_trial_status(subject, trial, dataset=dataset)
        path = paths.cached_trial_path(dataset, subject, trial)
        manifest = paths.read_manifest(path) or {}
        diagnostics = manifest.get('diagnostics') or {}
        rows.append({
            'dataset': dataset, 'subject': subject, 'trial': trial,
            'status': status, 'reason': reason or '',
            'has_build_report': build_report_path(dataset, subject, trial).exists(),
            'n_plates': diagnostics.get('n_plates'),
            'n_frames': diagnostics.get('n_frames'),
            'duration_s': diagnostics.get('duration_s'),
            'sample_rate_hz': diagnostics.get('sample_rate_hz'),
            'n_suspect_plates': len(diagnostics.get('suspect') or []),
            'suspect_plates': ','.join(diagnostics.get('suspect') or []),
            'bytes': path.stat().st_size if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def plate_diagnostics(dataset: str) -> pd.DataFrame:
    """The per-plate numbers the manifest already holds, as a table.

    `trial_diagnostics` computes a gyro residual, an accelerometer norm and a magnetometer
    norm for every plate of every trial, and `build_index` reduced all of it to
    `n_suspect_plates` -- a count of how many crossed 25 deg/s. The distribution behind that
    threshold, and therefore any justification for it, was invisible.

    `acc_norm_static_median` is the calibration check nobody was looking at: it should sit near
    9.81 on any plate that ever holds still, and a systematic departure is a scale error that
    propagates into every acceleration comparison.

    THE MAGNETOMETER NORM IS DELIBERATELY NOT HERE. It used to be, and it was not a usable
    flag: ||mag|| is exported in units normalized to each sensor's own calibration field, so
    the 0.89 it read says more about where the sensors were calibrated than about the data.
    It groups by SENSOR (ICC 0.43) rather than by session (0.09) or by height above the lab
    floor (0.01), which is the opposite of what a local field anomaly would do. The magnetic
    field is analysed properly in experiments/global_assumptions.py; a per-plate median here
    only invited reading a calibration constant as a data defect.

    EVERY ROW CARRIES ITS TRIAL'S CACHE STATUS, because a manifest is read whether or not its
    artifact is current and a stale one is in whatever format the build that wrote it used.
    Mixing them silently is not hypothetical: 238 rows here come from 18 stale static-pose
    manifests with no `residual_fraction` at all, so pooling a fresh-only numerator over a
    fresh-plus-stale denominator understated the suspect rate. Callers filter on `fresh`.
    """
    rows = []
    for subject, trial in get_source(dataset).enumerate_trials():
        manifest = paths.read_manifest(
            paths.cached_trial_path(dataset, subject, trial)) or {}
        status, _ = cached_trial_status(subject, trial, dataset=dataset)
        plates = (manifest.get('diagnostics') or {}).get('plates') or {}
        for plate, stats in plates.items():
            row = {'dataset': dataset, 'subject': subject, 'trial': trial, 'plate': plate,
                   'status': status, 'fresh': status == 'fresh'}
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    row[key] = value
                elif isinstance(value, (list, tuple)) and len(value) == 3:
                    row[f'{key}_magnitude'] = float(np.linalg.norm(value))
            rows.append(row)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------- coverage

def coverage_table(dataset: str, reports: pd.DataFrame) -> pd.DataFrame:
    """Which sensor was present in which trial, and where it went missing if it was not.

    The build FAILS SOFT on an untracked segment, deliberately: IMoVE's treadmill trials drop
    whole marker groups -- s10's t2 has all four LTH markers absent for the entire take -- and
    a reader has to be able to say "not tracked here" without failing the trial around it.
    The cost is that a trial builds successfully with 8 plates instead of 15 and nothing says
    so. This is the table that says so.

    `expected` is the union of every plate name the dataset produced anywhere, which is
    empirical rather than hardcoded, so it stays correct for a dataset this module has never
    seen. The reason a sensor is missing is triangulated from the pairing step:

      no_mocap      an IMU file exists but its segment never reconstructed
      no_imu        the segment reconstructed but no sensor was mounted on it
      neither       absent on both sides
      trial_failed  the trial did not build at all, so nothing is known per sensor
    """
    plates_per_trial, expected = {}, set()
    for subject, trial in get_source(dataset).enumerate_trials():
        manifest = paths.read_manifest(
            paths.cached_trial_path(dataset, subject, trial)) or {}
        names = set(((manifest.get('diagnostics') or {}).get('plates') or {}).keys())
        plates_per_trial[(subject, trial)] = names
        expected |= names

    pairing = reports[reports.step == 'S3_pairing'] if not reports.empty else pd.DataFrame()
    unmatched = {}
    if not pairing.empty:
        for (subject, trial), group in pairing.groupby(['subject', 'trial']):
            unmatched[(subject, trial)] = {
                'imu': set(group[group.metric.str.startswith('unmatched_imu')].value_str
                           .dropna()),
                'world': set(group[group.metric.str.startswith('unmatched_world')].value_str
                             .dropna()),
            }

    rows = []
    for (subject, trial), present in plates_per_trial.items():
        gaps = unmatched.get((subject, trial), {'imu': set(), 'world': set()})
        for sensor in sorted(expected):
            if sensor in present:
                status = 'present'
            elif not present:
                status = 'trial_failed'
            elif sensor in gaps['imu']:
                status = 'no_mocap'
            elif sensor in gaps['world']:
                status = 'no_imu'
            else:
                status = 'neither'
            rows.append({'dataset': dataset, 'subject': subject, 'trial': trial,
                         'sensor': sensor, 'status': status,
                         'present': status == 'present'})
    return pd.DataFrame(rows)


def invalid_sections(dataset: str) -> pd.DataFrame:
    """Every run of untrustworthy ground truth, with where in the trial it sits.

    Read from the built parquets rather than from the build report, because the report records
    counts and this needs positions -- and only the `valid` column is loaded, so the cost is a
    column read rather than a trial load.

    WHERE a gap sits changes what it means. At the head or tail it is coverage: the mocap
    started late or stopped early, and the inertial record extends past it. In the middle it is
    occlusion or a fault, which is the kind that corrupts a joint angle mid-motion.
    """
    rows = []
    for subject, trial in get_source(dataset).enumerate_trials():
        path = paths.cached_trial_path(dataset, subject, trial)
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=['plate', 'valid'])
        for plate, group in frame.groupby('plate', observed=True):
            valid = group['valid'].to_numpy(dtype=bool)
            if valid.all():
                continue
            edges = np.flatnonzero(
                np.concatenate([[True], valid[1:] != valid[:-1], [True]]))
            for start, stop in zip(edges[:-1], edges[1:]):
                if valid[start]:
                    continue
                position = ('head' if start == 0 else
                            'tail' if stop == len(valid) else 'middle')
                rows.append({'dataset': dataset, 'subject': subject, 'trial': trial,
                             'plate': str(plate), 'start_index': int(start),
                             'length': int(stop - start), 'position': position,
                             'n_frames': int(len(valid)),
                             'short': bool(stop - start < SHORT_INVALID_RUN)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- per-step pivoting

def step_table(reports: pd.DataFrame, step: str) -> pd.DataFrame:
    """One step's long-form rows pivoted wide: one row per entity, one column per metric."""
    subset = reports[reports.step == step]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(index=['dataset', 'subject', 'trial', 'entity_kind', 'entity'],
                              columns='metric', values='value_num', aggfunc='first')
    return wide.reset_index().rename_axis(None, axis=1)


# -------------------------------------------------------------------------------------- ICC

def intraclass_correlation(frame: pd.DataFrame, value: str, group: List[str]) -> float:
    """Fraction of a metric's variance that is BETWEEN groups rather than within them.

    1.0 means the members of a group are identical, so they are one observation wearing
    several hats; 0.0 means they are independent. This is the measurement that decides the
    blocking, and it exists because the obvious unit -- the sensor -- is wrong for every
    reconstruction metric in IMoVE, where three sensors share one marker cluster.
    """
    values = frame[[*group, value]].dropna()
    if values[value].nunique() <= 1 or len(values) < 4:
        return float('nan')
    grouped = values.groupby(group)[value]
    counts = grouped.count()
    if len(counts) < 2 or counts.max() < 2:
        return float('nan')
    grand = values[value].mean()
    between = float((counts * (grouped.mean() - grand) ** 2).sum() / max(len(counts) - 1, 1))
    within = float(grouped.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
                   / max(len(values) - len(counts), 1))
    mean_count = float(counts.mean())
    denominator = between + (mean_count - 1) * within
    return float((between - within) / denominator) if denominator > 0 else float('nan')


def icc_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every metric's ICC across each candidate grouping, so the blocking is evidence-based.

    Published rather than folded silently into the analysis, the way plotting/utils.py
    publishes the ICCs behind DEFAULT_BLOCK_COLS. A reader who disagrees with the blocking can
    see exactly what it was derived from.
    """
    candidates = {
        'placement': ['dataset', 'subject', 'trial', 'segment'],
        'trial': ['dataset', 'subject', 'trial'],
        'session': ['dataset', 'subject'],
    }
    rows = []
    for step, table in tables.items():
        if table.empty or 'entity' not in table:
            continue
        table = table.copy()
        # A plate name is '<SEGMENT>_<placement>' in IMoVE and a bare segment in Al Borno,
        # so stripping the last underscore group gives the shared reconstruction's identity.
        table['segment'] = table['entity'].astype(str).str.rsplit('_', n=1).str[0]
        numeric = [c for c in table.columns
                   if c not in ('dataset', 'subject', 'trial', 'entity_kind', 'entity',
                                'segment') and pd.api.types.is_numeric_dtype(table[c])]
        for metric in numeric:
            for label, group in candidates.items():
                if not set(group).issubset(table.columns):
                    continue
                rows.append({'step': step, 'metric': metric, 'grouping': label,
                             'icc': intraclass_correlation(table, metric, group)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame['dependent'] = frame['icc'] >= ICC_DEPENDENT
    return frame


def warn_if_placements_split(icc: pd.DataFrame) -> List[str]:
    """Metrics whose three placements are identical, i.e. must not be counted separately.

    Mirrors plotting/utils._warn_if_sides_split. Silence here would mean an inflated n on
    precisely the reconstruction metrics the report leads with.
    """
    if icc.empty:
        return []
    offenders = icc[(icc.grouping == 'placement') & icc.dependent]
    return sorted(f"{row.step}.{row.metric}" for row in offenders.itertuples())


# ----------------------------------------------------------------------------------- pooling

def pooled_summary(table: pd.DataFrame, step: str) -> pd.DataFrame:
    """Quantiles per metric, aggregated at this step's blocking level.

    Blocked BEFORE pooling: a metric that is identical across a segment's three placements is
    averaged into one value first, so it contributes one observation rather than three.
    """
    if table.empty:
        return pd.DataFrame()
    block = [c for c in BLOCKING.get(step, ('dataset', 'subject', 'trial'))
             if c in table.columns]
    numeric = [c for c in table.columns
               if c not in ('dataset', 'subject', 'trial', 'entity_kind', 'entity')
               and pd.api.types.is_numeric_dtype(table[c])]
    if not numeric:
        return pd.DataFrame()
    blocked = table.groupby(block, dropna=False)[numeric].mean().reset_index()

    rows = []
    for metric in numeric:
        values = blocked[metric].dropna()
        if values.empty:
            continue
        entry = {'step': step, 'metric': metric, 'n': int(len(values)),
                 'mean': float(values.mean()), 'std': float(values.std(ddof=1))
                 if len(values) > 1 else 0.0}
        entry.update({f'q{int(q * 100):02d}': float(values.quantile(q)) for q in QUANTILES})
        rows.append(entry)
    return pd.DataFrame(rows)


def health_score(index: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """A triage ranking, and explicitly nothing more.

    Components are normalized to [0, 1] where 1 is worse and averaged with equal weight. Equal
    weight is a placeholder, not a claim: the plan's X4 fits weights from each metric's
    correlation with downstream joint-angle error, and until that runs there is no evidence
    for any other weighting. Printed WITH its components always, and never used as a gate.
    """
    frames = []
    if 'S2_reconstruction' in tables and not tables['S2_reconstruction'].empty:
        recon = tables['S2_reconstruction']
        if 'residual_median_mm' in recon:
            frames.append(recon.groupby(['subject', 'trial'])['residual_median_mm'].max()
                          .rename('worst_residual_mm'))
        if 'valid_fraction' in recon:
            frames.append((1.0 - recon.groupby(['subject', 'trial'])['valid_fraction'].min())
                          .rename('worst_invalid_fraction'))
    if 'S4_sync' in tables and not tables['S4_sync'].empty and \
            'lag_mad_s' in tables['S4_sync']:
        frames.append(tables['S4_sync'].groupby(['subject', 'trial'])['lag_mad_s'].max()
                      .rename('sync_mad_s'))
    if 'S6_timeline' in tables and not tables['S6_timeline'].empty and \
            'origin_spread_s' in tables['S6_timeline']:
        frames.append(tables['S6_timeline'].groupby(['subject', 'trial'])['origin_spread_s']
                      .max().rename('origin_spread_s'))
    # The residual after the rotation, NOT the spread of the rotations themselves. The spread
    # was the first thing here and it was the wrong quantity: sensors are genuinely mounted at
    # different angles on different segments, so most of that spread is a fact about the
    # hardware rather than about how well the alignment worked. The residual is the fit's own
    # error, and dividing it by the measured signal's RMS makes it comparable between a static
    # pose and a sprint -- an absolute deg/s residual is not.
    if 'S7_alignment' in tables and not tables['S7_alignment'].empty and \
            'residual_fraction_of_signal' in tables['S7_alignment']:
        frames.append(tables['S7_alignment']
                      .groupby(['subject', 'trial'])['residual_fraction_of_signal']
                      .max().rename('worst_alignment_residual_fraction'))

    if not frames:
        return pd.DataFrame()
    components = pd.concat(frames, axis=1).reset_index()
    merged = index.merge(components, on=['subject', 'trial'], how='left')

    columns = [c for c in components.columns if c not in ('subject', 'trial')]
    normalized = []
    for column in columns:
        values = merged[column].astype(float)
        span = values.max() - values.min()
        normalized.append((values - values.min()) / span if span > 0
                          else pd.Series(0.0, index=values.index))
    merged['health'] = pd.concat(normalized, axis=1).mean(axis=1)
    merged['n_suspect_plates'] = merged['n_suspect_plates'].fillna(0)
    return merged.sort_values('health', ascending=False)


# ------------------------------------------------------------------------------------ report

def _header(number: int, title: str, subtitle: str = "") -> None:
    print(f"\n{'=' * 78}\n{number}. {title}\n{'=' * 78}")
    if subtitle:
        print(subtitle)


def _report_signal_repair(tables: Dict[str, pd.DataFrame]) -> None:
    """What the parser had to throw away and reconstruct, per file.

    Two separate faults share this path. A dropped radio packet leaves a HOLE, which the
    counter locates and the spline fills at the right instant. A sample beyond the sensor's
    full scale is a corrupt export, which is dropped and reconstructed from its neighbours --
    36 of those are in this dataset, peaking at 1.15e5 m/s^2. Both are repairs, and a repaired
    sample is interpolation rather than measurement, so the count belongs in a table.
    """
    table = tables.get('S1_parse')
    if table is None or table.empty:
        return
    _header(7, "Signal repair", "packets lost, samples out of range, and rows reconstructed")
    for column, label in (('missing_samples', 'packets lost'),
                          ('n_out_of_range_acc', 'acc samples beyond full scale'),
                          ('n_out_of_range_gyro', 'gyro samples beyond full scale'),
                          ('n_non_finite_rows', 'non-finite rows'),
                          ('n_rows_dropped', 'rows dropped and reconstructed')):
        if column not in table:
            continue
        values = table[column].fillna(0)
        affected = int((values > 0).sum())
        print(f"  {label:34s} {int(values.sum()):8d} in {affected:4d} of "
              f"{len(values):4d} files")
    if 'raw_acc_abs_max' in table and 'acc_abs_max' in table:
        worst = table.nlargest(5, 'raw_acc_abs_max')[
            ['entity', 'raw_acc_abs_max', 'acc_abs_max']]
        if float(worst.raw_acc_abs_max.max()) > 0:
            print("\n  Largest raw accelerations, against what survived the bound check:")
            for row in worst.itertuples():
                print(f"    {row.entity[:48]:48s} {row.raw_acc_abs_max:12.1f} -> "
                      f"{row.acc_abs_max:8.2f} m/s^2")


def _report_decimation_cost(tables: Dict[str, pd.DataFrame]) -> None:
    """How much signal the trial rate threw away.

    Every trial runs at the SLOWEST rate any of its streams was recorded at, which for the
    17-sensor IMoVE sessions is 40 Hz and therefore a 20 Hz Nyquist. That is the right choice
    -- comparing two signals in different bands is worse -- but the cost was never quantified.

    Measured, it is close to zero: the IMU is never the decimated stream in either dataset,
    and the mocap loses 5e-9 to 2e-5 of its power because Motive's output is already smoothed
    well below 20 Hz. The band limit that matters is the 40 Hz RECORDING rate, not this step.
    `imu_rate_hz` in this table is what says which trials have it.
    """
    table = tables.get('S5_resampling')
    if table is None or table.empty:
        return
    _header(8, "Decimation cost", "what the trial rate discarded, and where")
    if 'target_rate_hz' in table:
        # Rounded before counting. The rate is derived from a median sample interval, so a
        # 40 Hz trial can land on 39.99999999999999 and print as a second, identical-looking
        # "40.0" row -- which reads as two populations where there is one.
        rates = table.target_rate_hz.dropna().round(6)
        if not rates.empty:
            print("  trials by target rate:")
            print(rates.value_counts().sort_index().to_string())
    for column in ('acc_power_above_target_nyquist', 'gyro_power_above_target_nyquist',
                   'mocap_position_power_above_target_nyquist'):
        if column not in table:
            continue
        values = table[column].dropna()
        if values.empty:
            continue
        print(f"\n  {column:42s} n={len(values):5d} "
              f"q50 {100 * values.quantile(0.5):6.3f}%  "
              f"q95 {100 * values.quantile(0.95):6.3f}%  "
              f"max {100 * values.max():6.3f}%")
        worst = table.nlargest(5, column)[['entity', column]]
        for row in worst.itertuples():
            print(f"    {row.entity[:40]:40s} {100 * getattr(row, column):6.3f}% discarded")


def run(dataset: str, only_tables: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Build every table, print the console report, and return the tables."""
    wanted = set(only_tables) if only_tables else set(TRIAL_TABLES)

    index = build_index(dataset)
    reports, unavailable = load_build_reports(dataset)

    _header(0, "Coverage", f"{dataset}: {len(index)} trials enumerated")
    print(index.status.value_counts().to_string())
    print(f"\nbuild reports available for {int(index.has_build_report.sum())} of {len(index)}")
    if unavailable:
        print(f"  {len(unavailable)} trials have no tier-1 sidecar — built before the "
              f"instrumentation, or not rebuilt since. Not staleness: the report is not part "
              f"of the cache key.")
    if reports.empty:
        print("\nNo build reports at all. Rebuild with experiments/build_trials.py to collect "
              "them, then re-run.")
        return {'index': index}

    # How far each build got. For a trial with a parquet this is just the last step that had
    # anything to say; for one that FAILED it is the diagnostic -- the sidecar is written from
    # the failure path, so the last step with rows is where the exception came from.
    furthest = (reports.groupby(['subject', 'trial'])['step'].max()
                .rename('last_step_reported').reset_index())
    index = index.merge(furthest, on=['subject', 'trial'], how='left')
    failed = index[index.status.isin(('missing', 'failed'))
                   & index.last_step_reported.notna()]
    if not failed.empty:
        print(f"\n{len(failed)} trials with no artifact left a partial report:")
        print(failed.groupby('last_step_reported').size().to_string())

    coverage = coverage_table(dataset, reports)
    diagnostics = plate_diagnostics(dataset)
    sections = invalid_sections(dataset)

    _header(1, "Coverage", "which sensors are in which trials, and where they went missing")
    if not coverage.empty:
        print(coverage.status.value_counts().to_string())
        absent = coverage[~coverage.present]
        if not absent.empty:
            print(f"\n{len(absent)} sensor-trials absent. By sensor:")
            print(absent.groupby('sensor').size().sort_values(ascending=False)
                  .head(10).to_string())
            worst = (coverage.groupby(['subject', 'trial'])['present']
                     .agg(['sum', 'size']).sort_values('sum').head(8))
            print("\nLeast-covered trials (sensors present / expected):")
            for (subject, trial), row in worst.iterrows():
                missing = sorted(absent[(absent.subject == subject) &
                                        (absent.trial == trial)].sensor)
                print(f"  {subject}/{trial:26s} {int(row['sum'])}/{int(row['size'])}"
                      f"   missing {', '.join(m for m in missing[:6])}"
                      f"{' ...' if len(missing) > 6 else ''}")

    _header(2, "Invalid ground truth", "runs of untrustworthy pose, and where they sit")
    if not sections.empty:
        print(sections.groupby('position')['length'].describe()[['count', '50%', 'max']]
              .to_string())
        print(f"\n{int((~sections.short).sum())} runs are longer than "
              f"{SHORT_INVALID_RUN} frames; {int(sections.short.sum())} are blinks.")
        middle = sections[(sections.position == 'middle') & (~sections.short)]
        if not middle.empty:
            print(f"\n{len(middle)} MID-TRIAL gaps — the kind that corrupts a joint angle "
                  f"mid-motion rather than trimming an edge:")
            worst = middle.nlargest(8, 'length')
            for row in worst.itertuples():
                print(f"  {row.subject}/{row.trial}/{row.plate}: {row.length} frames "
                      f"at index {row.start_index} of {row.n_frames}")
    else:
        print("no invalid runs anywhere — every plate's pose is trustworthy throughout")

    tables = {step: step_table(reports, step) for step in STEPS}
    tables = {step: table for step, table in tables.items() if not table.empty}

    _header(3, "Per-step metrics", "pooled at each step's blocked level; see icc_report")
    summaries = []
    for number, (step, table) in enumerate(sorted(tables.items()), start=1):
        summary = pooled_summary(table, step)
        if summary.empty:
            continue
        summaries.append(summary)
        print(f"\n--- {step}  ({len(table)} entities, blocked to n={summary.n.max()})")
        print(summary[['metric', 'n', 'q50', 'q95', 'q99']].to_string(index=False))

    icc = icc_report(tables)
    _header(4, "Replicate structure", "measured ICCs, and the blocking they imply")
    if not icc.empty:
        print(icc.groupby('grouping')['icc'].describe()[['count', '50%', 'max']].to_string())
        split = warn_if_placements_split(icc)
        if split:
            print(f"\n{len(split)} metrics are IDENTICAL across a segment's placements — "
                  f"counting them per sensor would inflate n threefold:")
            for name in split[:8]:
                print(f"    {name}")

    if not diagnostics.empty:
        _header(6, "Per-plate diagnostics",
                "the distribution behind RESIDUAL_WARN_FRACTION, and the calibration checks")
        # FRESH ONLY. A stale manifest is in whatever format the build that wrote it used, so
        # pooling it here mixes populations and, worse, divides a numerator that only fresh
        # rows can contribute to by a denominator that includes rows which cannot.
        stale = diagnostics[~diagnostics.get('fresh', True)] if 'fresh' in diagnostics \
            else diagnostics.iloc[:0]
        current = diagnostics[diagnostics['fresh']] if 'fresh' in diagnostics else diagnostics
        if len(stale):
            # Counted on the PAIR: every one of these is named t0_static_pose_001, so counting
            # distinct trial names reports 18 stale trials as 1.
            n_stale = len(stale.groupby(['subject', 'trial']))
            print(f"  ({len(stale)} plates from {n_stale} stale trials excluded "
                  f"— their manifests predate these metrics)")
        for column, target in (('residual_fraction', None),
                               ('gyro_residual_lowpass_rms_deg_s', None),
                               ('acc_norm_static_median', 9.81),
                               ('acc_scale_error', 0.0)):
            if column not in current:
                continue
            values = current[column].dropna()
            if values.empty:
                continue
            print(f"  {column:34s} n={len(values):5d} "
                  f"q50 {values.quantile(0.5):7.2f}  q95 {values.quantile(0.95):7.2f}  "
                  f"max {values.max():8.2f}" + (f"   (expect ~{target})" if target else ""))
        if 'acc_norm_static_median' in current:
            never_still = int(current['acc_norm_static_median'].isna().sum())
            if never_still:
                print(f"  {never_still} plates never hold still, so they carry no evidence "
                      f"about their own accelerometer scale and report none")
        if 'residual_fraction' in current:
            scored = current['residual_fraction'].dropna()
            over = scored > RESIDUAL_WARN_FRACTION
            print(f"\n  {int(over.sum())} of {len(scored)} plates exceed "
                  f"{RESIDUAL_WARN_FRACTION:.0%} of their own signal "
                  f"({100 * over.mean():.1f}%)")
            # The check that the flag is no longer a speed detector. If a future change
            # reintroduces the dependence this line is where it shows.
            if 'gyro_signal_lowpass_rms_deg_s' in current:
                pair = current[['residual_fraction', 'gyro_signal_lowpass_rms_deg_s',
                                'gyro_residual_lowpass_rms_deg_s']].dropna()
                print(f"  correlation with plate speed: "
                      f"normalized r={pair.residual_fraction.corr(pair.gyro_signal_lowpass_rms_deg_s):+.2f}, "
                      f"absolute r={pair.gyro_residual_lowpass_rms_deg_s.corr(pair.gyro_signal_lowpass_rms_deg_s):+.2f}")

    _report_signal_repair(tables)
    _report_decimation_cost(tables)

    health = health_score(index, tables)
    _header(5, "Trial health", "a triage ranking, not a gate — components shown beside it")
    if not health.empty:
        columns = ['subject', 'trial', 'health'] + \
                  [c for c in health.columns if c.endswith(('_mm', '_s', '_deg', '_fraction'))]
        print(health[columns].head(15).to_string(index=False))

    written = {'index': index, 'report_rows': reports, 'icc_report': icc, 'health': health,
               'coverage': coverage, 'invalid_sections': sections,
               'plate_diagnostics': diagnostics}
    written.update({step.split('_', 1)[1]: table for step, table in tables.items()})
    if summaries:
        written['summary'] = pd.concat(summaries, ignore_index=True)

    for name, frame in written.items():
        if frame is not None and not frame.empty and (name in wanted or 'summary' == name):
            _save(frame, name, dataset)
    print(f"\nTables written to {_dataset_dir(dataset).relative_to(paths.REPO_ROOT)}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=TRIAL_DATASET)
    parser.add_argument('--only-tables', nargs='+', default=None,
                        help=f"Subset of {TRIAL_TABLES}")
    args = parser.parse_args()
    run(args.dataset, args.only_tables)


if __name__ == '__main__':
    main()
