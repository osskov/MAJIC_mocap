"""
How good is the joint centre — every dataset, one page.

`experiments/joint_center.py` measures this thoroughly and prints fourteen numbered sections to
the terminal, one dataset per invocation. That is the right shape for interrogating a result and
the wrong shape for finding out where you stand: the answer to "can I trust the joint centres on
this dataset, and how much error do they hand the projection" is spread over two terminal
scrollbacks that nobody kept.

This is the summary that survives the run. It reads ONLY the tables that experiment already
wrote, reduces each dataset to the handful of numbers a downstream user actually needs, and puts
every dataset in one file so they can be compared without re-running anything.

    python -m plotting.joint_center_quality                        # every measured dataset
    python -m plotting.joint_center_quality --datasets alborno

Writes results/reports/joint_center_quality.md.

IT IS A SUMMARY, NOT A REPLACEMENT. Nothing here is computed that the experiment does not already
compute, and every section names the deep-dive section it condenses, so a number that looks wrong
is traceable to the table and the prose behind it:

    python -m experiments.joint_center --dataset <name> --report-only

WHY CLOSURE LEADS. Two segments spanning a joint share one physical point, and what a relative
filter needs is that both segments construct the SAME point — not that the point is anatomically
the joint. So the measure of an offset pair is the RMS distance between the two segments'
constructions of it, in millimetres, and any six numbers can be scored that way on any trial. A
mocap fit, an anatomical midpoint and a gyro-only fit therefore land in one column, which is what
makes a cross-dataset table possible at all.

WHAT THIS DELIBERATELY DOES NOT DO is rank the datasets. Their trials differ in length, task and
ground-truth quality, and several of the metrics here move with all three — the holdout ratio
most of all. The table puts them side by side because that is useful; reading a winner out of it
is not supported, and the closing section says why per metric.
"""
import argparse
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import pipeline_constants
from experiments.global_assumptions import DATASETS, DatasetSpec, enumerate_trials, get_dataset
from experiments.joint_center import (MIN_FIT_FRAMES, SOURCE_ORDER, dataset_dir,
                                      load_trial_table, offset_stability)

REPORT_PATH = paths.REPO_ROOT / 'results' / 'reports' / 'joint_center_quality.md'

# What each closure source is, in one line, for the reader who meets the headline table first.
# Keyed by the labels `closure_by_source` writes, and the order they are reported in is
# SOURCE_ORDER, imported rather than repeated so the two cannot drift.
SOURCE_MEANING = {
    'own': 'fitted on this trial and scored on it. THE FLOOR, not a result — it is how badly a '
           'fixed-centre ball joint describes the pair, which no choice of offsets can remove.',
    'own_holdout': 'fitted on the first half of the trial, scored on the second. The cheapest '
                   'honest number.',
    'cross_trial': 'fitted on the subject\'s OTHER trials only. What you get by calibrating once '
                   'and reusing it.',
    'marker': 'an anatomical centre from markers, where the dataset has one. Not truth — the '
              'landmark carries its own placement error.',
    'inertial': 'the optimised Seel fit from gyroscope and accelerometer alone. The one row that '
                'says whether this works without a lab.',
    'inertial_ship': 'the shipped inertial estimator, for the same comparison.',
}

# The per-joint columns, in the order they are read: what the fit is, then how good it is, then
# how much room the metric leaves. Each entry is (header, unit, source column, format).
JOINT_COLUMNS = (
    ('lever arm', 'mm', 'arm_mm', '{:.0f}'),
    ('closure, own', 'mm', 'closure_own_mm', '{:.1f}'),
    ('held out', 'mm', 'closure_holdout_mm', '{:.1f}'),
    ('cross-trial', 'mm', 'closure_cross_mm', '{:.1f}'),
    ('between-trial spread', 'mm', 'spread_mm', '{:.1f}'),
    ('ambiguity, worst', 'mm', 'ambiguity_worst_mm', '{:.0f}'),
    ('excitation', '-', 'excitation', '{:.4f}'),
)


# ==============================================================================
# Loading
# ==============================================================================

def load_dataset(dataset: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Every joint_center table for one dataset, or None if it was never measured.

    None rather than empty frames, because "this dataset has no joint-centre analysis" and "this
    dataset was analysed and came out empty" are different findings and the coverage section
    reports them differently. A dataset whose directory exists but holds no fits counts as
    unmeasured: that is what a run which failed on every trial leaves behind.
    """
    fits = load_trial_table(dataset, 'joint_fits') if dataset_dir(dataset).is_dir() \
        else pd.DataFrame()
    if fits.empty:
        return None

    # The landmark and learning-curve tables are PROBED, NOT LOADED. This report only says whether
    # each exists, and concatenating them costs one parquet read per trial per table — 1100 reads
    # and most of the runtime on the larger datasets, for two booleans. `rglob` short-circuits on
    # the first hit, and it has to be recursive because a trial KEY may contain slashes: the
    # biplane half names its trials 'Test1/A/RSDrop1', so those tables sit three levels down.
    tables = {'fits': fits, 'stability': offset_stability(fits),
              'has_landmarks': next(dataset_dir(dataset).rglob('landmarks.parquet'), None)
                               is not None,
              'has_curve': next(dataset_dir(dataset).rglob('learning_curve.parquet'), None)
                           is not None}
    # The three dataset-level tables are written by the experiment's summary stage rather than
    # per trial, so a run stopped before it is missing them; absent is empty, never an error.
    for name in ('closure_by_source', 'ambiguity', 'subject_fits'):
        path = dataset_dir(dataset) / f'{name}.parquet'
        tables[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return tables


@lru_cache(maxsize=None)
def built_trial_count(dataset: str) -> int:
    """How many trials are BUILT for a dataset, for the coverage denominator.

    Cached because `enumerate_trials` walks the build tree and cross-checks it against the source
    enumeration, which costs 13-23 s per dataset here — four calls dominated this report's runtime,
    and the tests, which render the report repeatedly, spent almost all their time on repeated
    identical walks. Keyed by name rather than by build tree: `imove_biplane_vicon` shares
    `imove_biplane`'s trials, and reporting each spec's own count is the honest denominator for a
    row about that spec.
    """
    return len(enumerate_trials(dataset))


def primary(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Rows for the dataset's PRIMARY joints only.

    IMoVE carries its High and Low sensor placements as extra joint pairs — 18 pairs for 6 joints
    — and pooling them into a headline would report the same anatomical joint three times and
    weight that dataset triple. The variants are a real result and they belong in
    `experiments.sensor_placement`, not in a summary of how good the joint centre is.
    """
    if frame.empty or 'joint' not in frame:
        return frame
    return frame[frame['joint'].isin(spec.primary_joints)]


# ==============================================================================
# Per-dataset reduction
# ==============================================================================

def per_joint(tables: Dict[str, pd.DataFrame], spec: DatasetSpec) -> pd.DataFrame:
    """One row per primary joint: the numbers to quote when using that joint's offsets.

    Every column is a MEAN OVER TRIALS, because a trial's fit is the unit of observation here —
    each number in `joint_fits` is already a reduction over one trial's frames — and the between-
    trial spread beside it is what says how much that mean can be trusted.
    """
    fits = primary(tables['fits'], spec)
    fits = fits[fits['converged']] if not fits.empty else fits
    if fits.empty:
        # Empty rather than a table of NaNs: a dataset that fitted nothing has no per-joint row to
        # show, and the coverage section has already said why in words.
        return pd.DataFrame()

    grouped = fits.groupby('joint', observed=True)
    out = pd.DataFrame({
        'n_trials': grouped.size(),
        'arm_mm': grouped[['parent_norm_mm', 'child_norm_mm']].mean().mean(axis=1),
        'excitation': grouped['excitation'].mean(),
        'gain': grouped['gain_median'].mean(),
        'unmasked_shift_mm': grouped['unmasked_shift_mm'].mean(),
    })

    closure = primary(tables['closure_by_source'], spec)
    for source, column in (('own', 'closure_own_mm'), ('own_holdout', 'closure_holdout_mm'),
                           ('cross_trial', 'closure_cross_mm')):
        rows = closure[closure['source'] == source] if not closure.empty else closure
        out[column] = (rows.groupby('joint', observed=True)['closure_mm'].mean()
                       if not rows.empty else np.nan)

    # The error bar to quote, and the one the cost budget uses: the worse of the two segments'
    # between-trial wander, not the in-trial residual. The residual is a model mismatch the fit
    # has already absorbed; this is whether the same six numbers come back next time.
    stability = primary(tables['stability'], spec)
    if not stability.empty:
        spread = stability.groupby('joint', observed=True)[
            ['parent_spread_mm', 'child_spread_mm']].mean().max(axis=1)
        out['spread_mm'] = spread
        out['n_subjects_paired'] = stability.groupby('joint', observed=True).size()
    else:
        out['spread_mm'] = np.nan
        out['n_subjects_paired'] = 0

    ambiguity = primary(tables['ambiguity'], spec)
    out['ambiguity_worst_mm'] = (ambiguity.groupby('joint', observed=True)['worst_mm'].mean()
                                 if not ambiguity.empty else np.nan)

    # The downstream translation: |da| <= (|alpha| + |omega|^2)|dr|, gain measured per trial.
    # An UPPER BOUND — the two terms are perpendicular to different things and partly cancel —
    # so it is a budget, not an estimate.
    out['acc_error'] = out['gain'] * out['spread_mm'] / 1000.0
    out['acc_error_vs_std'] = out['acc_error'] / pipeline_constants()['acc_std']
    return out.reindex([j for j in spec.primary_joints if j in out.index])


def closure_summary(tables: Dict[str, pd.DataFrame], spec: DatasetSpec) -> pd.DataFrame:
    """Median closure per source, pooled over the dataset's primary joint-trials.

    MEDIAN rather than mean: the inertial sources have a long right tail — a fit that fails
    outright lands hundreds of millimetres out — and a mean there describes the failures, not the
    method. The per-joint table carries means, where the distribution is tight enough for one.
    """
    closure = primary(tables['closure_by_source'], spec)
    if closure.empty:
        return pd.DataFrame()
    grouped = closure.groupby('source', observed=True)['closure_mm']
    out = pd.DataFrame({'n': grouped.size(), 'median_mm': grouped.median(),
                        'p90_mm': grouped.quantile(0.90)})
    floor = out['median_mm'].get('own', np.nan)
    out['vs_floor'] = out['median_mm'] / floor
    offsets = closure.groupby('source', observed=True)['offset_from_own_mm'].median()
    out['offset_from_own_mm'] = offsets
    return out.reindex([s for s in SOURCE_ORDER if s in out.index])


# ==============================================================================
# Markdown
# ==============================================================================

def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    """A markdown table, or nothing at all when there are no rows.

    Returning nothing for an empty body matters: a header with no rows under it reads as "measured
    and came out empty" rather than "not measured", and the coverage section is the only place
    allowed to make that distinction.
    """
    if not rows:
        return []
    return (['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
            + ['| ' + ' | '.join(cells) + ' |' for cells in rows] + [''])


def _fmt(value, template: str = '{:.1f}', dash: str = '—') -> str:
    """A number, or an em dash where there is none. Never 'nan' in a report someone reads."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return template.format(value)


def coverage_section(measured: Dict[str, Dict[str, pd.DataFrame]],
                     requested: Optional[List[str]] = None) -> List[str]:
    """Which datasets have a joint-centre analysis, which do not, and how to fix that.

    FIRST SECTION AND LOAD-BEARING. The question that sent someone to this file is usually "is my
    dataset covered", and a report that answers it only implicitly — by that dataset's absence
    from a table further down — is how an unmeasured dataset gets mistaken for a measured one
    with nothing to report. Every REGISTERED dataset appears here, including the ones with
    nothing, each with the command that would fill it in.
    """
    requested = sorted(DATASETS) if requested is None else requested
    lines = ['## Which datasets have been measured', '',
             'Every dataset registered in `experiments/global_assumptions.py` is listed, measured '
             'or not. `joint-trials` counts (trial x joint) fits that converged over the primary '
             f'joints; a trial with fewer than {MIN_FIT_FRAMES} valid frames on a pair is not '
             'fitted at all and contributes none.', '']

    rows = []
    for name in sorted(DATASETS):
        spec = get_dataset(name)
        built = built_trial_count(name)
        tables = measured.get(name)
        if tables is None:
            # NOT REQUESTED IS NOT THE SAME AS NOT MEASURED, and conflating them turns a narrowed
            # run into a false claim about the rest of the repository. Only a dataset this run
            # actually looked for on disk can be reported as absent from it.
            status = '**no**' if name in requested else 'not checked'
            rows.append([f'`{name}`', str(built), status, '—', '—', '—', '—'])
            continue
        fits = primary(tables['fits'], spec)[lambda f: f['converged']]
        trials = fits[['subject', 'trial']].drop_duplicates()
        closure = primary(tables['closure_by_source'], spec)
        sources = set(closure['source']) if not closure.empty else set()
        arms = [s for s in SOURCE_ORDER if s in sources and s != 'own']
        rows.append([f'`{name}`', str(built), 'yes' if len(fits) else 'ran, **0 fits**',
                     str(trials['subject'].nunique()) if len(fits) else '—',
                     f"{len(trials)}" if len(fits) else '0',
                     f"{len(fits)} over {fits['joint'].nunique()} joints" if len(fits) else '0',
                     ', '.join(f'`{a}`' for a in arms) or '—'])

    lines += _table(['dataset', 'built trials', 'measured', 'subjects', 'trials fitted',
                     'joint-trials', 'comparison arms present'], rows)

    # WHICH ARMS ARE MISSING AND WHY, derived from disk rather than asserted. An absent arm is
    # ambiguous between "this dataset cannot support it" and "the run that would have produced it
    # has not happened", and only the second is actionable — so each case is named separately.
    for name, tables in measured.items():
        spec = get_dataset(name)
        rows_all = primary(tables['fits'], spec)
        closure = primary(tables['closure_by_source'], spec)
        sources = set(closure['source']) if not closure.empty else set()

        # A DATASET THAT RAN AND FITTED NOTHING IS THE MOST MISREADABLE ROW IN THE TABLE, because
        # it looks like a failure and is not one. Say what the floor was and what the data had, so
        # the reader can see it is a property of the recordings rather than of the code.
        if rows_all.empty or not rows_all['converged'].any():
            valid = rows_all['n_valid'].median() if 'n_valid' in rows_all else np.nan
            lines += [f"> `{name}` ran over every trial and fitted NOTHING, which is the correct "
                      f"answer rather than a failure. A fit needs {MIN_FIT_FRAMES} frames on which "
                      f"both segments have trustworthy ground truth, and the median joint-trial "
                      f"here has {_fmt(valid, '{:.0f}')}. That is a property of the recordings — "
                      f"fluoroscopy images a fraction of a second — and no amount of re-running "
                      f"changes it. Where the same recordings carry a longer-lived reference, "
                      f"that spec is the one to read.", '']
            continue

        if 'marker' not in sources and tables['has_landmarks']:
            lines += [f"> `{name}` has a landmark table on disk but no `marker` arm in its "
                      f"closure table, so the stored table predates that arm. Re-running the "
                      f"experiment adds it — the marker arm is the only one that reads the source "
                      f"`.trc`/CSV, and it is the one an earlier `--only-tables` run would have "
                      f"skipped.", '']
        if 'inertial' not in sources:
            lines += [f"> `{name}` has no `inertial` arm, which means "
                      f"`experiments/inertial_joint_center.py` has not been run on it. That is "
                      f"the only row that says whether the joint centre is recoverable without "
                      f"mocap, so its absence is a gap in the argument and not just in the "
                      f"table:", '', '```bash',
                      f'python -m experiments.inertial_joint_center --dataset {name}', '```', '']

    unmeasured = [name for name in requested if name not in measured]
    if unmeasured:
        lines += [f"{len(unmeasured)} registered dataset(s) carry built trials but no "
                  f"joint-centre analysis. To measure one:", '',
                  '```bash', f"python -m experiments.joint_center --dataset {unmeasured[0]}",
                  '```', '',
                  'That is a real gap and not a formatting one — nothing downstream of the '
                  'projection has an error bar on those datasets\' offsets. Note that a dataset '
                  'whose ground truth is valid for only a fraction of a second per trial cannot '
                  f'clear the {MIN_FIT_FRAMES}-frame floor and will report zero fits, which is '
                  'the correct answer rather than a failure.', '']
    return lines


def headline_section(measured: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
    """The cross-dataset closure table — the reason this file exists in one piece."""
    lines = ['## The headline: closure, scored out of sample', '',
             'Closure is the RMS distance between the two segments\' constructions of the joint, '
             'in millimetres:', '',
             '    RMS_t | (p_parent + R_parent c_parent) - (p_child + R_child c_child) |', '',
             'Any six numbers can be scored on any trial, so every source below is in the same '
             'units and comparable across datasets. Medians over primary joint-trials; `vs floor` '
             'is the ratio to that dataset\'s own in-sample fit.', '']

    rows, present, mismatched = [], set(), []
    for name, tables in measured.items():
        spec = get_dataset(name)
        summary = closure_summary(tables, spec)
        present |= set(summary.index)
        for source, row in summary.iterrows():
            rows.append([f'`{name}`', f'`{source}`', str(int(row['n'])),
                         _fmt(row['median_mm'], '{:.1f}'), _fmt(row['p90_mm'], '{:.1f}'),
                         _fmt(row['vs_floor'], '{:.2f}x'),
                         _fmt(row['offset_from_own_mm'], '{:.0f}')])
        # `own` is one row per joint-trial with stored normal equations; the per-joint table below
        # counts CONVERGED joint_fits rows. They should agree, and where they do not the reason is
        # that the two tables come from different runs — worth saying rather than leaving a reader
        # to notice two totals that differ by six and wonder which is wrong.
        fitted = len(primary(tables['fits'], spec).query('converged'))
        if 'own' in summary.index and int(summary.loc['own', 'n']) != fitted:
            mismatched.append(f"`{name}` ({int(summary.loc['own', 'n'])} vs {fitted})")
    lines += _table(['dataset', 'source', 'n', 'closure (mm)', 'p90 (mm)', 'vs floor',
                     'offset from own (mm)'], rows)
    if mismatched:
        lines += [f"> `n` on the `own` row counts joint-trials with stored normal equations, and "
                  f"the per-joint table below counts converged `joint_fits` rows. They disagree on "
                  f"{', '.join(mismatched)}, which means those two tables were written by "
                  f"different runs — re-run the experiment on that dataset to bring them back "
                  f"into step.", '']

    lines += ['What each source is:', '']
    lines += [f'- **`{source}`** — {SOURCE_MEANING[source]}'
              for source in SOURCE_ORDER if source in present]
    lines += ['',
              'READ THE LAST COLUMN AGAINST THE THIRD, because it is where this metric is most '
              'often misread. A source can sit tens of millimetres from the trial\'s own offsets '
              'and still score within a millimetre of it. That is not a contradiction: closure is '
              'flat along the direction those two fits differ in, and the per-joint table\'s '
              '`ambiguity, worst` column says how flat, in the same millimetres.', '',
              'AND THAT FLATNESS IS NOT A DEFECT. Closure is blind to a simultaneous slide of '
              'both offsets along the joint axis, and so is the application — two segments '
              'constructing the same point are both computing the acceleration of that one point '
              'whether or not it is anatomically the joint. The metric\'s null space and the '
              'projection\'s null space coincide.', '']
    return lines


def per_joint_section(measured: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
    """One table per dataset: the joint-by-joint answer to "how good is this one"."""
    lines = ['## Per joint', '',
             'The numbers to quote when a downstream experiment uses one joint\'s offsets. Means '
             'over that joint\'s trials.', '']
    for name, tables in measured.items():
        spec = get_dataset(name)
        table = per_joint(tables, spec)
        if table.empty:
            continue
        rows = [[joint, str(int(row['n_trials']))]
                + [_fmt(row[column], template) for _, _, column, template in JOINT_COLUMNS]
                for joint, row in table.iterrows()]
        lines += [f'### `{name}`', '']
        lines += _table(['joint', 'trials'] + [f'{h} ({u})' if u != '-' else h
                                               for h, u, _, _ in JOINT_COLUMNS], rows)
        paired = int(table['n_subjects_paired'].sum())
        if paired == 0:
            lines += ['> No subject has two or more trials on any joint here, so there is no '
                      'between-trial spread and no error bar. Every number in this table '
                      'describes single fits.', '']
    lines += ['**lever arm** is the mean of the two offset magnitudes — the distance '
              '`IMUTrace.project_acc` moves the accelerometer along, and what an offset error '
              'gets multiplied by. Read closure against it: the ratio is the fraction of the '
              'projection distance the joint model cannot account for.', '',
              '**between-trial spread** is how far a subject\'s fitted offset moves between their '
              'own trials. The offset is anatomy plus mounting, both constant within a session, '
              'so this is fit error — and note how much larger it is than `closure, own`. The '
              'residual says how well six parameters fit one trial; this says whether they are '
              'the SAME six parameters next time.', '',
              '**ambiguity, worst** is the displacement costing one millimetre of closure, along '
              'the worst-determined direction. It is the tolerance the metric grants you, in the '
              'units the offsets are quoted in, and it is usually far larger than the spread '
              'beside it. **excitation** is what sets it: 0 when the two segments never move '
              'relative to each other, growing as the joint sweeps SO(3). A pure hinge leaves the '
              'offset along its axis unobservable however long the trial runs.', '']
    return lines


def cost_section(measured: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
    """What the offset error costs the acceleration projection — the reason to care."""
    acc_std = pipeline_constants()['acc_std']
    lines = ['## What an offset error costs downstream', '',
             'The translation into the currency `experiments/acceleration_projection.py` works '
             'in. `project_acc` adds `alpha x r + omega x (omega x r)`, both linear in `r`, so an '
             'offset error `dr` propagates with a gain of at most `|alpha| + |omega|^2` — measured '
             'from this trial\'s own gyroscope and its derivative, not from a nominal. The offset '
             'error used is the between-trial spread, not the in-trial residual.', '']
    rows = []
    for name, tables in measured.items():
        table = per_joint(tables, get_dataset(name))
        for joint, row in table.iterrows():
            rows.append([f'`{name}`', joint, _fmt(row['gain'], '{:.1f}'),
                         _fmt(row['spread_mm'], '{:.1f}'), _fmt(row['acc_error'], '{:.3f}'),
                         _fmt(row['acc_error_vs_std'], '{:.0f}x')])
    lines += _table(['dataset', 'joint', 'gain (1/s^2)', 'offset error (mm)',
                     'acc error (m/s^2)', f'vs acc_std ({acc_std:g})'], rows)
    lines += ['It is an UPPER BOUND — the two terms are perpendicular to different things and '
              'partially cancel for any given `dr` — so read it as a budget, not an estimate. The '
              'last column is that budget against the accelerometer standard deviation the filter '
              'is tuned with, which is what says whether the joint-centre fit is a limiting error '
              'source or a rounding one.', '']
    return lines


def caveats_section(measured: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
    """The ways these numbers get over-read, each named with the column it applies to."""
    lines = ['## How to read this, and what it does not say', '']
    lines += ['- **`own` is not an accuracy.** It is in sample, and six free parameters reduce any '
              'residual. What it measures is how badly a fixed-centre ball joint describes the '
              'pair: soft tissue, a knee that translates as it flexes, marker reconstruction '
              'error and residual sensor-to-segment misalignment all land in it.',
              '- **Do not rank the datasets on `held out`.** A holdout split in TIME assumes the '
              'two halves sample the same joint behaviour, and in a short single-task recording '
              'they plainly do not — the first half is one posture and the second another. That '
              'ratio moves with trial length and homogeneity, so it is comparable only between '
              'datasets whose trials are alike.',
              '- **Closure locates nothing anatomically.** A source can close perfectly with both '
              'offsets slid along the joint axis. Where an anatomical position is what you need — '
              'a segment length, a landmark comparison — closure is the wrong metric and the '
              'experiment\'s sections 8 and 13 are the right ones.',
              '- **Nothing here is validated against truth.** The `marker` arm is an independent '
              'estimate carrying its own error, and the largest part of that error — the marker '
              'having been taped somewhere slightly wrong — is invisible to any scatter, because '
              'it sits constant. Agreement between the two is mutual support, not proof that '
              'either is right.',
              '- **The offsets are not read from these tables.** Downstream consumers call '
              '`joint_center.joint_offsets()`, which refits from the built trial at ~17 ms per '
              'joint. This report describes those fits; it is not the artifact they come from, so '
              'it cannot go stale against them in a way that changes a result.']
    variants = [name for name, tables in measured.items()
                if tables['fits']['joint'].nunique() > len(get_dataset(name).primary_joints)]
    if variants:
        lines.append(f"- **Placement variants are excluded from every table above.** "
                     f"{', '.join(f'`{v}`' for v in variants)} carries more than one sensor per "
                     f"segment, so the same anatomical joint is fitted from several placements. "
                     f"Pooling them would report that joint repeatedly and weight the dataset "
                     f"accordingly. How much the estimate depends on which sensor you fit from is "
                     f"`experiments.sensor_placement`.")
    lines.append('')
    return lines


def provenance_section(measured: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
    """Every file behind this report, and the commands that regenerate them."""
    lines = ['## Where these numbers come from', '',
             'Every table read by this report, all written by '
             '`experiments/joint_center.py`. Each carries a `.manifest.json` sidecar with the git '
             'SHA, the constants in force and the command line that produced it — `results/` is '
             'not in version control, so that sidecar is the only record of which code version '
             'wrote the file beside it.', '']
    lines += _table(
        ['path', 'what it holds'],
        [['`results/experiments/joint_center/<dataset>/<subject>/<trial>/joint_fits.parquet`',
          'per-joint scalars: offsets, residual, conditioning, precision, migration, gain'],
         ['`.../normal_equations.parquet`',
          'the 6x6 Gram matrix per joint — every closure and ambiguity number is derived from '
          'these without reloading a trial'],
         ['`.../landmarks.parquet`', 'the fitted offset against an anatomical centre'],
         ['`.../learning_curve.parquet`', 'offset error against fit length'],
         ['`results/experiments/joint_center/<dataset>/closure_by_source.parquet`',
          'the headline table: every candidate offset pair scored on one metric'],
         ['`.../ambiguity.parquet`', 'the displacement costing 1 mm of closure, per direction'],
         ['`.../subject_fits.parquet`', 'one fit per subject instead of one per trial, and its '
                                       'cross-validation'],
         ['`.../segment_lengths.parquet`', 'the physical invariant the fit has to preserve'],
         ['`results/statistics/joint_center_<dataset>_statistics.parquet`',
          'per-(subject, joint) quantiles of each scalar over trials']])

    lines += ['To regenerate everything behind this file:', '', '```bash']
    for name in measured:
        lines.append(f'python -m experiments.joint_center --dataset {name}')
    lines += ['python -m plotting.joint_center_quality', '```', '',
              'The full deep dive — fourteen sections, per dataset, to the terminal — is the same '
              'experiment with `--report-only`, which recomputes the summary from what is already '
              'on disk and loads no trials. For the geometric sanity check the numbers cannot '
              'give you (a knee centre floating in front of the shank, a sensor rotated against '
              'its neighbours) it is `python -m plotting.joint_center --dataset <name>`.', '']

    extras = []
    for name, tables in measured.items():
        if not tables['has_landmarks']:
            extras.append(f'`{name}` has no landmark comparison on disk')
        if not tables['has_curve']:
            extras.append(f'`{name}` has no learning curve on disk')
    if extras:
        lines += ['Two per-trial tables are optional and absent for some datasets: '
                  + '; '.join(extras) + '. They are produced by the same run — '
                  '`--only-tables` narrows which tables a run writes, so a narrowed earlier run '
                  'is the usual reason one is missing.', '']
    return lines


def write_report(measured: Dict[str, Dict[str, pd.DataFrame]], output: Path = REPORT_PATH,
                 requested: Optional[List[str]] = None) -> Path:
    """The report. Markdown because it is diffable: comparing this file across a pipeline change
    is exactly the review you want, and `results/` being gitignored means keeping a copy across
    one means copying it elsewhere first."""
    total_joint_trials = sum(len(primary(t['fits'], get_dataset(n))) for n, t in measured.items())
    lines = [
        '# Joint centre estimation quality', '',
        f'{len(measured)} of {len(DATASETS)} registered datasets measured · '
        f'{total_joint_trials} primary joint-trials fitted', '',
        'Two segments spanning a joint share one physical point. Fitting where that point sits in '
        'each segment\'s own frame gives the six numbers `IMUTrace.project_acc` moves the '
        'accelerometer along and the magnetometer projection extrapolates the field across, so '
        'every projection in this repository inherits whatever error is in them. This is that '
        'error, per dataset.', '',
        '**The number to quote** is the between-trial spread in the per-joint table — how far a '
        'subject\'s fitted offset moves between their own trials — not the in-trial residual. The '
        'residual is a model mismatch the fit has already absorbed into the offsets it returns; '
        'the spread is whether the same six numbers come back next time, which is what a '
        'downstream experiment using one trial\'s offsets is exposed to.', '',
        'Condensed from `experiments/joint_center.py`, which computes all of this and prints '
        'fourteen sections per dataset. Nothing here is recomputed: every number traces to a '
        'table on disk, and the last section says which.', '',
    ]
    lines += coverage_section(measured, requested)
    lines += headline_section(measured)
    lines += per_joint_section(measured)
    lines += cost_section(measured)
    lines += caveats_section(measured)
    lines += provenance_section(measured)

    path = paths.ensure_parent(output)
    path.write_text('\n'.join(lines))
    # Relative where it can be — the repo-relative path is the clickable one — but an --output
    # anywhere on disk is legitimate and must not crash the write that already succeeded.
    try:
        shown = path.relative_to(paths.REPO_ROOT)
    except ValueError:
        shown = path
    print(f"Wrote report to {shown}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=sorted(DATASETS), choices=sorted(DATASETS),
                        metavar='DATASET',
                        help="Datasets to include. Defaults to every registered one; those with "
                             "no analysis on disk are listed as unmeasured rather than skipped.")
    parser.add_argument('--output', type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    measured = {}
    for name in args.datasets:
        tables = load_dataset(name)
        if tables is None:
            print(f"{name}: no joint-centre analysis on disk.")
            continue
        measured[name] = tables
        print(f"{name}: {len(tables['fits'])} joint-trials.")

    if not measured:
        print("\nNothing measured on any requested dataset. Run, for example:\n"
              f"  python -m experiments.joint_center --dataset {args.datasets[0]}")
        return 1
    write_report(measured, args.output, args.datasets)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
