"""Figures for the build-quality analysis, and the markdown report that carries them.

Reads ONLY the parquet tables written by experiments/build_quality.py. Nothing here reloads a
trial or re-derives a statistic, so no number in a figure can disagree with the table beside
it -- the same rule plotting/acceleration_projection.py follows, and it is the reason the
figures can be re-tuned without a rebuild.

ORGANIZED BY ARGUMENT, not by pipeline step. Build-quality quantities span millimetres to
degrees to seconds, and putting them on shared axes because they happen to come from
consecutive steps is the main way to misread the result. The step tag survives as a table
column, so both organizations remain available.

  Is the ground truth trustworthy?   reconstruction residuals, validity, coherence of the
                                     mocap-derived signal with the measured one
  Did the build damage anything?     per-plate sync agreement, timeline consistency,
                                     alignment residual structure
  Is the artifact faithful?          what survived serialization and resampling

WHAT IS NOT HERE YET, and why it is worth knowing: no figure claims a significance test. The
plan routes everything through plot_utils.test_panels with Friedman then Holm-corrected
Wilcoxon, and that needs the control variants (§3.4) -- the same quantity measured with a
build step disabled -- which the instrumentation does not yet produce. Drawing brackets from
the blocked tables alone would be testing a difference between metrics rather than between
treatments, which is not a claim worth making. The blocking IS applied to the pooled numbers.

    python -m plotting.build_quality --dataset imove
    python -m plotting.build_quality --dataset imove --report-only
"""
import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from experiments.build_quality import EXPERIMENT_NAME, ICC_DEPENDENT
from plotting.utils import finalize_and_save_plot

PLOTS_SUBDIR = 'build_quality'

# Tolerances the build actually uses, drawn as reference lines so a residual is read against
# the threshold that governed it rather than against an eyeballed scale.
CLUSTER_TOLERANCE_MM = 10.0
FOOT_TOLERANCE_MM = 25.0
RESIDUAL_WARN_DEG_S = 25.0


def _dataset_dir(dataset: str) -> Path:
    return paths.experiment_dir(EXPERIMENT_NAME) / dataset


def load_tables(dataset: str) -> Dict[str, pd.DataFrame]:
    """Every table the experiment wrote. Missing ones are simply absent, not fatal --
    --only-tables exists precisely so a targeted rerun is cheap."""
    tables = {}
    for path in sorted(_dataset_dir(dataset).glob('*.parquet')):
        tables[path.stem] = pd.read_parquet(path)
    return tables


def _plots_dir(dataset: str) -> Path:
    return paths.PLOTS_DIR / PLOTS_SUBDIR / dataset


def _coverage(tables: Dict[str, pd.DataFrame]) -> str:
    """Footer naming what is pooled, so a figure cannot be read as covering more than it does."""
    index = tables.get('index')
    if index is None or index.empty:
        return ""
    built = int((index.status == 'fresh').sum())
    reported = int(index.has_build_report.sum()) if 'has_build_report' in index else 0
    return (f"{reported} of {len(index)} trials instrumented, {built} fresh · "
            f"{index.subject.nunique()} sessions")


# ------------------------------------------------------------------ is the truth trustworthy?

def plot_reconstruction(tables: Dict[str, pd.DataFrame], dataset: str, save: bool,
                        show: bool) -> None:
    """Residual ECDFs per segment kind, against the two tolerances the build applies.

    An ECDF rather than a box, because the argument is in the TAIL: the reason the reader
    carries two tolerances is that rigid clusters and deforming foot markers are two
    populations, and a box plot's whiskers cut off exactly the part that shows it.
    """
    recon = tables.get('reconstruction')
    if recon is None or recon.empty or 'residual_median_mm' not in recon:
        return

    recon = recon.copy()
    recon['kind'] = np.where(recon.entity.astype(str).str.contains('FOOT'), 'foot',
                             'cluster')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for kind, group in recon.groupby('kind'):
        values = np.sort(group['residual_median_mm'].dropna().to_numpy())
        if not len(values):
            continue
        axes[0].step(values, np.arange(1, len(values) + 1) / len(values), where='post',
                     label=f'{kind} (n={len(values)})', linewidth=2)
    for level, label in ((CLUSTER_TOLERANCE_MM, 'cluster tol'),
                         (FOOT_TOLERANCE_MM, 'foot tol')):
        axes[0].axvline(level, color='grey', linestyle='--', linewidth=1)
        axes[0].text(level, 0.02, f' {label}', fontsize=8, color='grey', rotation=90)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('median fit residual (mm)')
    axes[0].set_ylabel('fraction of segment-trials')
    axes[0].set_title('Reconstruction residual, by segment kind')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    if 'valid_fraction' in recon:
        order = recon.groupby('entity')['valid_fraction'].median().sort_values().index[:20]
        subset = recon[recon.entity.isin(order)]
        axes[1].boxplot([subset[subset.entity == name]['valid_fraction'].dropna()
                         for name in order],
                        labels=[str(name).split('/')[-1] for name in order],
                        vert=False, showfliers=False,
                        whis=(5, 95))          # p5/p95, matching the house _box_stats
        axes[1].set_xlabel('valid fraction')
        axes[1].set_title('Least-complete segments (p5–p95 whiskers)')
        axes[1].grid(alpha=0.3, axis='x')
        axes[1].tick_params(labelsize=7)

    finalize_and_save_plot(fig, f'Ground-truth quality — {dataset}',
                           'reconstruction.png', plots_dir=_plots_dir(dataset),
                           epilog=_coverage(tables), save=save, show=show)


# ------------------------------------------------------------------- did the build damage it?

def plot_sync_and_timeline(tables: Dict[str, pd.DataFrame], dataset: str, save: bool,
                           show: bool) -> None:
    """Per-plate sync agreement and the timeline spread, both of which the build reduces to
    one number per trial.

    The right panel is the s16 defect as a routine measurement: when a trial's plates disagree
    about when ground truth starts, the late ones spend that long on a held pose while sharing
    the others' clock. It was invisible until the origin became a trial-level quantity.
    """
    sync, timeline = tables.get('sync'), tables.get('timeline')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if sync is not None and not sync.empty and 'deviation_from_median_s' in sync:
        per_trial = sync.dropna(subset=['deviation_from_median_s'])
        order = (per_trial.groupby(['subject', 'trial'])['deviation_from_median_s']
                 .apply(lambda s: float(np.abs(s).max())).sort_values(ascending=False))
        top = order.head(25).index
        for position, key in enumerate(top):
            values = per_trial.set_index(['subject', 'trial']).loc[[key],
                                                                   'deviation_from_median_s']
            axes[0].scatter(np.abs(values), np.full(len(values), position), s=12, alpha=0.7)
        axes[0].set_yticks(range(len(top)))
        axes[0].set_yticklabels([f'{s}/{t}'.replace('_001', '') for s, t in top], fontsize=6)
        axes[0].set_xscale('symlog', linthresh=1e-3)
        axes[0].set_xlabel('|per-plate lag − trial median| (s)')
        axes[0].set_title('Worst 25 trials by sync disagreement')
        axes[0].grid(alpha=0.3, axis='x')

    if timeline is not None and not timeline.empty and 'deviation_from_origin_s' in timeline:
        values = timeline['deviation_from_origin_s'].dropna()
        nonzero = values[values > 1e-9]
        axes[1].hist(np.concatenate([[0.0], nonzero]) if len(nonzero) else values,
                     bins=40, color='#4477aa', edgecolor='white')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('plate first-valid − trial origin (s)')
        axes[1].set_ylabel('plates (log)')
        axes[1].set_title(f'Timeline agreement — {len(nonzero)} of {len(values)} plates '
                          f'start late')
        axes[1].grid(alpha=0.3, axis='y')

    finalize_and_save_plot(fig, f'Synchronization and timeline — {dataset}',
                           'sync_timeline.png', plots_dir=_plots_dir(dataset),
                           epilog=_coverage(tables), save=save, show=show)


def plot_alignment(tables: Dict[str, pd.DataFrame], dataset: str, save: bool,
                   show: bool) -> None:
    """The sensor-to-segment rotation, which the build solves and has never reported.

    The left panel is the conditioning caveat that RESIDUAL_WARN_DEG_S cannot express: a plate
    with few valid frames yields a small residual and a meaningless rotation, and the absolute
    residual alone cannot tell that apart from a well-aligned plate.

    The right panel asks whether the rotations are a mounting convention or a fit that found
    something else -- an offset near a coordinate axis is a sensor clipped in the wrong way
    round, an arbitrary direction is not.
    """
    alignment = tables.get('alignment')
    if alignment is None or alignment.empty or 'offset_angle_deg' not in alignment:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if 'n_frames_used' in alignment:
        axes[0].scatter(alignment['n_frames_used'], alignment['offset_angle_deg'],
                        s=10, alpha=0.4, color='#4477aa')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('valid frames the rotation was fitted on (log)')
        axes[0].set_ylabel('sensor-to-segment offset (deg)')
        axes[0].set_title('Alignment vs how much data supported it')
        axes[0].grid(alpha=0.3)

    if 'angle_to_nearest_plate_axis_deg' in alignment:
        axes[1].hist(alignment['angle_to_nearest_plate_axis_deg'].dropna(), bins=40,
                     color='#ee6677', edgecolor='white')
        axes[1].set_xlabel('angle from rotation axis to nearest coordinate axis (deg)')
        axes[1].set_ylabel('plates')
        axes[1].set_title('Mounting convention, or an arbitrary fit?')
        axes[1].grid(alpha=0.3, axis='y')

    finalize_and_save_plot(fig, f'Sensor-to-segment alignment — {dataset}',
                           'alignment.png', plots_dir=_plots_dir(dataset),
                           epilog=_coverage(tables), save=save, show=show)


def plot_replicate_structure(tables: Dict[str, pd.DataFrame], dataset: str, save: bool,
                             show: bool) -> None:
    """The measured ICCs behind the blocking.

    Published as a figure for the same reason plotting/utils.py publishes the ICCs behind
    DEFAULT_BLOCK_COLS: the blocking changes every n in the report, so a reader has to be able
    to see what it was derived from rather than take it on trust.
    """
    icc = tables.get('icc_report')
    if icc is None or icc.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for grouping, group in icc.groupby('grouping'):
        values = np.sort(group['icc'].dropna().to_numpy())
        if not len(values):
            continue
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post',
                label=f'{grouping} (n={len(values)})', linewidth=2)
    ax.axvline(ICC_DEPENDENT, color='crimson', linestyle='--', linewidth=1.2)
    ax.text(ICC_DEPENDENT, 0.05, ' not independent →', color='crimson', fontsize=8)
    ax.set_xlabel('intraclass correlation')
    ax.set_ylabel('fraction of metrics')
    ax.set_title('How much of each metric is shared within a grouping')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    finalize_and_save_plot(fig, f'Replicate structure — {dataset}',
                           'replicate_structure.png', plots_dir=_plots_dir(dataset),
                           epilog=_coverage(tables), save=save, show=show)


def plot_health(tables: Dict[str, pd.DataFrame], dataset: str, save: bool,
                show: bool) -> None:
    """Trial ranking with its components beside it, never the score alone.

    The components are the point. A composite is a triage aid, and one shown without what went
    into it invites being read as a verdict.
    """
    health = tables.get('health')
    if health is None or health.empty or 'health' not in health:
        return

    components = [c for c in health.columns
                  if c.endswith(('_mm', '_s', '_deg', '_fraction')) and c != 'duration_s']
    worst = health.dropna(subset=['health']).head(20)
    if worst.empty:
        return

    fig, axes = plt.subplots(1, 1 + len(components),
                            figsize=(4 + 2.4 * len(components), 6), sharey=True)
    axes = np.atleast_1d(axes)
    labels = [f'{row.subject}/{row.trial}'.replace('_001', '')
              for row in worst.itertuples()]
    positions = np.arange(len(worst))

    axes[0].barh(positions, worst['health'], color='#333333')
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('composite')
    axes[0].set_title('health (triage only)', fontsize=10)
    axes[0].grid(alpha=0.3, axis='x')

    for axis, column in zip(axes[1:], components):
        axis.barh(positions, worst[column].astype(float), color='#4477aa')
        axis.set_xlabel(column, fontsize=8)
        axis.grid(alpha=0.3, axis='x')
        axis.tick_params(labelsize=7)

    finalize_and_save_plot(fig, f'Worst trials, with components — {dataset}',
                           'health.png', plots_dir=_plots_dir(dataset),
                           epilog=_coverage(tables), save=save, show=show)


# ------------------------------------------------------------------------------ the deliverable

def write_report(tables: Dict[str, pd.DataFrame], dataset: str) -> Path:
    """The markdown report.

    Markdown rather than HTML because it diffs in git: a diff of this file after a pipeline
    change is exactly the review you want, and is the reason it is committed while the parquets
    are not.
    """
    index = tables.get('index', pd.DataFrame())
    summary = tables.get('summary', pd.DataFrame())
    icc = tables.get('icc_report', pd.DataFrame())
    health = tables.get('health', pd.DataFrame())

    lines = [f'# Build quality — {dataset}', '']
    if not index.empty:
        counts = index.status.value_counts().to_dict()
        instrumented = int(index.has_build_report.sum())
        lines += [
            f'{len(index)} trials enumerated · '
            + ' · '.join(f'{n} {status}' for status, n in sorted(counts.items())),
            f'{instrumented} instrumented with a build report · '
            f'{int(index.n_suspect_plates.fillna(0).sum())} suspect plate-flags',
            '',
        ]
        if instrumented < len(index):
            lines += [
                f'> {len(index) - instrumented} trials carry no build report. That is not '
                f'staleness — the report is not part of the cache key, so a trial built '
                f'before the instrumentation existed is still valid and simply has no '
                f'tier-1 data. Rebuild to fill them in.', '']

    if not health.empty and 'health' in health:
        ranked = health.dropna(subset=['health']).head(3)
        if not ranked.empty:
            lines += ['## Three things to look at', '']
            for row in ranked.itertuples():
                lines.append(f'- **{row.subject}/{row.trial}** — composite '
                             f'{row.health:.2f}')
            lines.append('')

    coverage = tables.get('coverage', pd.DataFrame())
    sections = tables.get('invalid_sections', pd.DataFrame())

    if not coverage.empty:
        lines += ['## What is included, and what is not', '',
                  'The build FAILS SOFT on a segment it cannot track, which is deliberate — '
                  'IMoVE’s treadmill trials drop whole marker groups, and a reader has to be '
                  'able to distinguish "not tracked here" from "this trial failed". The cost '
                  'is that a trial builds successfully with fewer plates than expected and '
                  'nothing says so. This is where it says so.', '',
                  'Every row below is a sensor that the rest of the dataset has and this trial '
                  'does not, so `expected` is empirical rather than a hardcoded roster.', '']
        counts = coverage.status.value_counts()
        lines += ['| status | sensor-trials | meaning |', '| --- | --- | --- |']
        meaning = {
            'present': 'built into the artifact',
            'no_mocap': 'IMU recorded, but its segment never reconstructed',
            'no_imu': 'segment reconstructed, but no sensor was mounted on it',
            'neither': 'absent on both sides — e.g. the 7-sensor long-walk sessions, which '
                       'genuinely have no High or Low placements',
            'trial_failed': 'the trial did not build, so nothing is known per sensor',
        }
        for status, n in counts.items():
            lines.append(f'| `{status}` | {n} | {meaning.get(status, "")} |')
        lines.append('')

        absent = coverage[~coverage.present]
        if not absent.empty:
            by_sensor = absent.groupby('sensor').size().sort_values(ascending=False)
            lines += ['### Sensors most often missing', '',
                      '| sensor | absent in N trials |', '| --- | --- |']
            lines += [f'| `{name}` | {n} |' for name, n in by_sensor.head(12).items()]
            lines.append('')

            per_trial = (coverage.groupby(['subject', 'trial'])['present']
                         .agg(present='sum', expected='size').reset_index()
                         .sort_values('present'))
            partial = per_trial[per_trial.present < per_trial.expected]
            lines += [f'### Trials with incomplete coverage ({len(partial)} of '
                      f'{len(per_trial)})', '',
                      '| subject | trial | present / expected | missing |',
                      '| --- | --- | --- | --- |']
            for row in partial.head(25).itertuples():
                missing = sorted(absent[(absent.subject == row.subject) &
                                        (absent.trial == row.trial)].sensor)
                shown = ', '.join(f'`{m}`' for m in missing[:5])
                more = f' +{len(missing) - 5}' if len(missing) > 5 else ''
                lines.append(f'| {row.subject} | {row.trial} | '
                             f'{row.present} / {row.expected} | {shown}{more} |')
            lines.append('')

    if not sections.empty:
        lines += ['## Where ground truth is untrustworthy', '',
                  'A frame marked invalid stays on the uniform time grid so filters and finite '
                  'differences still work; the mask keeps it out of error statistics. WHERE a '
                  'run sits changes what it means — at an edge it is coverage, in the middle it '
                  'is occlusion or a structural gap.', '']
        # Not named `median` or `max`: those collide with Series methods, so `row.median`
        # silently returns the bound method and formatting it raises rather than printing a
        # wrong number. Loud, but only because the name was chosen badly.
        by_position = sections.groupby('position')['length'].agg(
            runs='size', median_length='median', longest_run='max')
        lines += ['| position | runs | median length | longest | meaning |',
                  '| --- | --- | --- | --- | --- |']
        position_meaning = {
            'head': 'mocap started after the inertial record',
            'tail': 'mocap stopped before it',
            'middle': 'occlusion, a fault, or a structural gap between mocap takes',
        }
        for position, row in by_position.iterrows():
            lines.append(f'| {position} | {int(row.runs)} | {row.median_length:.0f} | '
                         f'{int(row.longest_run)} | {position_meaning.get(position, "")} |')
        lines.append('')

        middle = sections[(sections.position == 'middle') & (~sections.short)]
        if not middle.empty:
            lines += [f'### Longest mid-trial gaps ({len(middle)} runs)', '',
                      'The long-walk sessions dominate here by design: one inertial record '
                      'spans three mocap takes, so the gaps between takes are real absences of '
                      'ground truth rather than faults.', '',
                      '| subject | trial | plate | frames | at index | of |',
                      '| --- | --- | --- | --- | --- | --- |']
            for row in middle.nlargest(15, 'length').itertuples():
                lines.append(f'| {row.subject} | {row.trial} | `{row.plate}` | '
                             f'{row.length} | {row.start_index} | {row.n_frames} |')
            lines.append('')

    if not summary.empty:
        lines += ['## By processing step', '']
        for step, group in summary.groupby('step'):
            lines += [f'### {step}', '',
                      '| metric | n | median | p95 | p99 |',
                      '| --- | --- | --- | --- | --- |']
            for row in group.sort_values('metric').itertuples():
                lines.append(f'| {row.metric} | {row.n} | {row.q50:.4g} | '
                             f'{row.q95:.4g} | {row.q99:.4g} |')
            lines.append('')

    if not icc.empty:
        dependent = icc[(icc.grouping == 'placement') & icc.dependent]
        lines += ['## Replicate structure', '',
                  'Measured intraclass correlations decide the blocking rather than an '
                  'assumption about what an independent observation is.', '']
        table = icc.groupby('grouping')['icc'].median().round(3)
        lines += ['| grouping | median ICC |', '| --- | --- |']
        lines += [f'| {name} | {value} |' for name, value in table.items()]
        lines += ['']
        if not dependent.empty:
            lines += [f'{len(dependent)} metrics are effectively identical across a segment’s '
                      f'placements (ICC ≥ {ICC_DEPENDENT}), so counting them per sensor would '
                      f'inflate n threefold:', '']
            lines += [f'- `{row.step}.{row.metric}`' for row in dependent.itertuples()]
            lines += ['']

    lines += ['## Not yet covered', '',
              'No figure here carries a significance test. Doing that properly needs the '
              'control variants — the same quantity measured with a build step disabled — '
              'which the instrumentation does not yet produce. Brackets drawn from these '
              'tables alone would compare metrics rather than treatments.', '',
              'Also absent: raw-parse statistics (including the PacketCounter gap check), '
              'spectra and coherence, serialization fidelity, and the predictive-value '
              'ranking against downstream joint-angle error.', '']

    path = paths.ensure_parent(paths.REPO_ROOT / 'results' / 'reports' /
                               f'build_quality_{dataset}.md')
    path.write_text('\n'.join(lines))
    print(f"Wrote report to {path.relative_to(paths.REPO_ROOT)}")
    return path


FIGURES = (plot_reconstruction, plot_sync_and_timeline, plot_alignment,
           plot_replicate_structure, plot_health)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='imove')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()

    tables = load_tables(args.dataset)
    if not tables:
        print(f"No tables for {args.dataset}. Run experiments/build_quality.py first.")
        return

    if not args.report_only:
        for figure in FIGURES:
            figure(tables, args.dataset, save=not args.no_save, show=args.show)
    write_report(tables, args.dataset)


if __name__ == '__main__':
    main()
