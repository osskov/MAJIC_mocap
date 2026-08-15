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
import re
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


# One caption per figure, keyed by filename, so the text rendered onto the PNG and the text
# in the markdown report cannot drift apart. Each says what is plotted, how to read it, and
# what it does NOT show -- a figure travels into a slide or a message without the code or the
# report section that explains it, and arrives having to stand on its own.
CAPTIONS = {
    'data_state.png':
        'Triage view of the whole dataset: every session (rows) against every activity '
        '(columns). LEFT: how many sensors each trial produced as a fraction of what that '
        'session can produce, with the raw count in the cell and the session expectation in '
        'the row label. The denominator is per session on purpose — the long-walk sessions '
        'carry 7 sensors rather than 15, and scoring them against the dataset-wide roster '
        'would show complete sessions as half broken. MIDDLE: the composite health score — '
        'the equally-weighted mean of five build diagnostics, each rescaled to [0, 1] across '
        'this dataset: the worst segment\'s marker-fit residual (mm), the largest fraction of '
        'frames with untrustworthy ground truth, how much the trial\'s plates disagreed about '
        'the IMU-to-mocap lag (s), how far apart the plates think the trial starts (s), and '
        'the spread in fitted sensor-to-segment angle across plates (deg). Because the '
        'rescaling is within-dataset, the darkest cell is the worst trial HERE and not a bad '
        'trial in absolute terms; a dataset with no problems would still have a darkest cell. '
        'See health.png for the components plotted separately. RIGHT: each session\'s mean '
        'health, the fastest way to see which sessions are more and less suspect. '
        'Hatched = the trial did not build. Grey = '
        'it built but scored nothing, which is every static-pose trial, having no motion for '
        'the sync and alignment components to measure. White = no such trial in that '
        'session. Blocks of missing coverage are protocol, not fault: the treadmill '
        'activities instrument the right leg only. NOT SHOWN: anything about accuracy — a '
        'fully covered, low-health trial can still carry a systematic error, since every '
        'component here is internal consistency rather than agreement with a reference.',
    'reconstruction.png':
        'Whether the marker-derived ground truth can be trusted, before any comparison with '
        'an IMU. A segment\'s pose is recovered by fitting a rigid template to its markers, '
        'and both panels measure how well that fit went. LEFT: empirical CDF (y = fraction of '
        'segment-trials at or below x) of `median fit residual`, the median over frames of '
        'the root-mean-square distance in MILLIMETRES between where each marker was seen and '
        'where the rigid template puts it. Small means the segment really did move as one '
        'rigid body; large means the markers moved relative to each other, so the derived '
        'pose is not trustworthy. Log x. The dashed lines are the tolerances the build itself '
        'applied — 10 mm for a marker cluster, 25 mm for a foot — so a residual is read '
        'against the threshold that governed it rather than an eyeballed scale. Foot segments '
        'are drawn separately because they deform: their tolerance is looser by design and '
        'pooling them with clusters would make the distribution meaningless. RIGHT: the '
        'twenty segments with the lowest `valid fraction`, the proportion of frames whose '
        'residual came in under tolerance, i.e. the share of the trial where that segment\'s '
        'pose is usable at all. Whiskers at p5–p95. A low valid fraction and a low residual '
        'together mean the fit succeeded on the few frames it had, which is weaker evidence '
        'than a low residual alone suggests. NOT SHOWN: WHERE in a trial the invalid frames '
        'sit — an edge gap is coverage and a mid-trial gap corrupts a joint angle, and that '
        'distinction is in the invalid_sections table.',
    'sync_timeline.png':
        'Whether the build put a trial\'s plates on one clock. The IMU and the mocap are '
        'recorded on separate clocks, and the offset between them — the LAG — is estimated '
        'per plate by cross-correlating the measured gyroscope against the angular velocity '
        'differentiated from the marker-derived pose. Both panels show quantities the build '
        'otherwise reduces to a single number per trial, which is what hid them. LEFT: the '
        'twenty-five worst trials by sync disagreement. Each point is one plate\'s '
        '`|per-plate lag − trial median|` in SECONDS: how far that plate\'s lag estimate fell '
        'from the median of the trial\'s plates. All the plates were recorded in one session '
        'on one clock, so the true value is zero for every point — anything visible is the '
        'estimator failing, usually because the trial has too little motion for the '
        'cross-correlation to find a peak. Symlog x, so agreement at the microsecond level '
        'and disagreement at the second level are both readable. RIGHT: `plate first-valid − '
        'trial origin`, seconds between the trial\'s shared t = 0 and the first frame on '
        'which that plate has trustworthy ground truth; log counts on y. Zero is the '
        'expected value and the bar at zero holds most plates. A plate that starts late '
        'spends that long on a held pose while sharing the other plates\' clock, which is '
        'the s16 defect as a routine measurement. NOT SHOWN: clock DRIFT within a trial. '
        'Every lag here is one constant per plate, and the ~20 ppm relative drift measured '
        'over a long-walk record — about 11 ms over 540 s — is in none of these numbers.',
    'alignment.png':
        'The sensor-to-segment rotation, which the build solves for every plate and never '
        'reported until now. A sensor is mounted on its segment in an unknown orientation, so '
        'the build fits the single rotation that best carries the marker-derived angular '
        'velocity onto the measured gyroscope; both panels describe that fitted rotation, one '
        'point per plate. LEFT: `sensor-to-segment offset` in DEGREES, the total angle of '
        'that rotation, against `valid frames the rotation was fitted on`, the number of '
        'frames where both signals were trustworthy (log x). This is the conditioning caveat '
        'a residual threshold cannot express: a plate with few valid frames yields a small '
        'residual and a meaningless rotation, and the residual alone cannot tell that apart '
        'from a well-aligned plate. Read the left edge with suspicion regardless of where the '
        'point sits vertically. RIGHT: `angle from rotation axis to nearest coordinate axis` '
        'in degrees — treating the fitted rotation as an axis and an angle, how far that axis '
        'lies from the closest of the plate\'s own x, y, z. Zero means the sensor was rotated '
        'about one of the plate\'s own axes, which is what clipping a sensor on in one of a '
        'few fixed orientations produces. A pile-up near zero therefore means the rotations '
        'are a mounting convention; a flat spread means the fit found something that is not. '
        'NOT SHOWN: whether the rotation is CORRECT. Nothing here compares it to an '
        'independent measurement of how the sensor was actually mounted, because no such '
        'measurement exists in either dataset.',
    'replicate_structure.png':
        'The measured intraclass correlations behind the blocking, published rather than '
        'asserted because the blocking changes every n in the report. The INTRACLASS '
        'CORRELATION of a metric within a grouping is the share of that metric\'s total '
        'variance explained by which group a measurement belongs to: 0 means members of a '
        'group are as unlike each other as any two measurements, so each is a genuine '
        'replicate, and 1 means they are the same measurement recorded more than once. Each '
        'curve is the empirical CDF (y = fraction of metrics at or below x) of ICC across '
        'metrics for one candidate grouping — placement, segment, session and so on. The '
        f'dashed line at {ICC_DEPENDENT} is the threshold past which a grouping\'s members '
        'are pooled into one observation instead of counted separately. The case this exists '
        'to catch: IMoVE mounts three sensors (High, Mid, Low) on one segment and all three '
        'share a single marker reconstruction, so a reconstruction metric counted per sensor '
        'would treble-count one measurement and inflate every n threefold. A curve sitting '
        'far to the right of the line is exactly that. NOT SHOWN: any significance test. '
        'These ICCs set the blocking; they are not themselves a comparison between '
        'treatments.',
    'health.png':
        'The worst twenty trials by composite health, with every component that went into '
        'the score shown beside it — never the score alone. WHAT EACH PANEL PLOTS, left to '
        'right: `composite`, the equally-weighted mean of the five components after each is '
        'rescaled to [0, 1] across this dataset, so it is unitless and relative. '
        '`worst_residual_mm`, the largest median rigid-body fit residual among the trial\'s '
        'segments, in millimetres — how far the markers sat from the rigid shape the segment '
        'is assumed to be, for the worst segment in that trial. `worst_invalid_fraction`, the '
        'largest fraction of frames marked untrustworthy on any one segment, 0 to 1. '
        '`sync_mad_s`, the median absolute deviation in SECONDS of the per-plate IMU-to-mocap '
        'lag estimates within the trial — one recording session has one true lag, so this is '
        'how much the plates disagreed about it, and a large value means the estimator failed '
        'rather than that the clocks differ. `origin_spread_s`, seconds between the earliest '
        'and latest plate\'s first valid frame, i.e. how far apart the plates think the trial '
        'starts. `alignment_angle_spread_deg`, the standard deviation in degrees of the '
        'fitted sensor-to-segment offset angle across the trial\'s plates — sensors mounted '
        'the same way should agree, so spread is either genuine mounting variety or a fit '
        'that failed on some plates. The equal weighting is a placeholder rather than a '
        'claim: until the components are regressed against downstream joint-angle error '
        'there is no evidence for any other weighting. Read the component bars, not the '
        'composite — a trial can rank high on one component and be fine in every other '
        'respect — and note this is never used as a gate on the data. NOT SHOWN: absolute '
        'quality. Top of this figure means worst in this dataset, which in a clean dataset '
        'still means acceptable.',
}


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


# ----------------------------------------------------------------------- what state is it in?

def _natural_key(name: str):
    """Split digits from text so 's2' sorts before 's13' and 't2' before 't10'."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r'(\d+)', str(name))]


def _session_expectation(coverage: pd.DataFrame) -> pd.Series:
    """How many sensors each SESSION can produce, taken from the session itself.

    NOT the dataset-wide union, which is the trap this exists to avoid. IMoVE's five long-walk
    sessions carry 7 sensors at 100 Hz where the other 21 carry 15 at 40 Hz, so scoring every
    trial against 15 would paint s4l, s5l, s6l, s13l and s23l as 47% broken when they are
    complete. The per-session maximum is empirical and stays right for a dataset this module
    has never seen.

    The cost is that a session broken in EVERY trial looks whole, because its own maximum
    drops with it. `n_expected` is annotated on the figure for exactly that reason: a row
    reading 7 in a dataset whose others read 15 is a question, not a reassurance.
    """
    present = coverage[coverage.present].groupby(['subject', 'trial']).size()
    return present.groupby('subject').max()


def plot_data_state(tables: Dict[str, pd.DataFrame], dataset: str, save: bool,
                    show: bool) -> None:
    """The triage figure: what loaded, how completely, and which sessions look worst.

    Everything else in this file answers a question you already knew to ask. This one is for
    walking up to the dataset cold and seeing its shape -- which is otherwise spread across
    four tables and 262 rows.

    Two grids on one subject axis, so a row can be read straight across:

      COVERAGE   how much of the session each trial actually produced. A trial can build
                 successfully with 8 plates instead of 15 -- the build fails soft on an
                 untracked segment, deliberately -- and that is invisible in a status column.
      HEALTH     the composite from `health_score`, which is a TRIAGE RANKING and not a
                 verdict. Normalized within the dataset, so the darkest cell is the worst
                 trial here rather than a bad trial in absolute terms. A dataset with no
                 problems at all would still have a darkest cell.

    The two disagree usefully. A trial can be fully covered and still rank badly, which points
    at sync or alignment rather than at missing sensors; the reverse points at coverage.
    """
    index, coverage = tables.get('index'), tables.get('coverage')
    if index is None or index.empty or coverage is None or coverage.empty:
        return
    health = tables.get('health')

    subjects = sorted(index.subject.unique(), key=_natural_key)
    trials = sorted(index.trial.unique(), key=_natural_key)
    rows, columns = {s: i for i, s in enumerate(subjects)}, \
                    {t: i for i, t in enumerate(trials)}
    shape = (len(subjects), len(trials))

    expected = _session_expectation(coverage)
    n_present = coverage[coverage.present].groupby(['subject', 'trial']).size()

    # NaN means "no such trial in this session", which must not read as zero coverage: Al
    # Borno's 05, 08 and 10 have walking only, and IMoVE's sessions do not all run every
    # protocol. Left uncoloured rather than dark.
    fraction = np.full(shape, np.nan)
    counts = np.full(shape, np.nan)
    unbuilt = np.zeros(shape, dtype=bool)
    for row in index.itertuples():
        r, c = rows[row.subject], columns[row.trial]
        if row.status in ('missing', 'failed', 'absent'):
            unbuilt[r, c] = row.status != 'absent'
            continue
        count = float(n_present.get((row.subject, row.trial), 0))
        counts[r, c] = count
        fraction[r, c] = count / max(float(expected.get(row.subject, 1)), 1.0)

    score = np.full(shape, np.nan)
    if health is not None and not health.empty and 'health' in health:
        for row in health.dropna(subset=['health']).itertuples():
            score[rows[row.subject], columns[row.trial]] = row.health

    # Floored, because Al Borno has two trials against IMoVE's thirteen. Sized from the grid
    # alone the figure came out narrower than its own caption, and `bbox_inches='tight'` then
    # expanded the canvas to fit the text and left the panels crushed against one edge. The
    # marginal's share is tied to the grid width for the same reason.
    marginal = max(3.0, 0.4 * len(trials))
    figure, axes = plt.subplots(
        1, 3, figsize=(max(9.0, 3.0 + 0.68 * len(trials)), 2.5 + 0.22 * len(subjects)),
        gridspec_kw={'width_ratios': [len(trials), len(trials), marginal]}, sharey=True)

    # The composite is a mean of normalized components, so it never approaches 1 even for the
    # worst trial -- on IMoVE the maximum is around 0.5. Scaling the colour to 1 would render
    # the whole panel in the palest third of the map and hide every difference in it. Scaled
    # to the observed maximum instead, which is also what "worst here" already meant.
    worst = float(np.nanmax(score)) if np.isfinite(score).any() else 1.0
    labels = [t.replace('_001', '') for t in trials]
    for axis, values, cmap, top, title, bar_label in (
            (axes[0], fraction, 'YlGn', 1.0,
             'Coverage — produced / expected', 'fraction of the session'),
            (axes[1], score, 'YlOrRd', worst,
             'Health — triage only', f'composite (max {worst:.2f})')):
        image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=top,
                            aspect='auto', interpolation='nearest')
        axis.set_xticks(range(len(trials)))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(subjects)))
        # The sensor count rides on the tick label. A session reading 7 where its neighbours
        # read 15 is the one thing the per-session normalization can hide, so it is never off
        # the figure.
        axis.set_yticklabels([f'{s} ({int(expected.get(s, 0))})' for s in subjects],
                             fontsize=7)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(np.arange(-0.5, len(trials)), minor=True)
        axis.set_yticks(np.arange(-0.5, len(subjects)), minor=True)
        axis.grid(which='minor', color='white', linewidth=0.8)
        axis.tick_params(which='minor', length=0)
        figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01).set_label(
            bar_label, fontsize=7)

    # A trial that did not build is a different KIND of thing from one that built badly, so it
    # gets a mark rather than a place on the same scale. Hatching survives greyscale printing
    # and colour-vision deficiency, which a red cell does not.
    for r, c in zip(*np.nonzero(unbuilt)):
        for axis in axes[:2]:
            axis.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#f2f2f2',
                                         edgecolor='#cc3311', hatch='///', linewidth=0.8))

    # A BLANK IN THE HEALTH PANEL WOULD OTHERWISE BE AMBIGUOUS. Left alone it reads as "not in
    # that session", but a trial can also build cleanly and still score nothing, when none of
    # the components the composite is made of were recorded for it -- s13l's static pose is
    # one. Filled grey so the two cannot be confused: white means there is no such trial, grey
    # means there is one and it has not been scored.
    unscored = np.isnan(score) & ~np.isnan(counts)
    for r, c in zip(*np.nonzero(unscored)):
        axes[1].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#dddddd',
                                        edgecolor='white', linewidth=0.8))

    # The raw count, because the fraction alone cannot distinguish 7/7 from 15/15 -- and that
    # distinction is the whole reason _session_expectation is per session.
    for (r, c), count in np.ndenumerate(counts):
        if np.isnan(count):
            continue
        axes[0].text(c, r, f'{int(count)}', ha='center', va='center', fontsize=5.5,
                     color='#333333' if fraction[r, c] > 0.5 else '#777777')

    # Which sessions are more and less suspect, which is hard to read off a grid by eye.
    with np.errstate(invalid='ignore'):
        per_subject = np.nanmean(score, axis=1)
    axes[2].barh(np.arange(len(subjects)), np.nan_to_num(per_subject), color='#bb5566',
                 height=0.72)
    axes[2].set_title('Session mean', fontsize=10)
    axes[2].set_xlabel('mean health', fontsize=8)
    axes[2].grid(alpha=0.3, axis='x')
    axes[2].tick_params(labelsize=7)
    axes[0].invert_yaxis()

    axes[0].set_xlabel('cell = sensors produced · row label = what the session expects',
                       fontsize=7)
    epilog = (f"{_coverage(tables)} · hatched = did not build ({int(unbuilt.sum())}) · "
              f"grey = built, unscored ({int(unscored.sum())}) · white = not in that session")
    finalize_and_save_plot(figure, f'What loaded, and what looks suspect — {dataset}',
                           'data_state.png', plots_dir=_plots_dir(dataset),
                           caption=CAPTIONS['data_state.png'], epilog=epilog, save=save, show=show)


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
                           caption=CAPTIONS['reconstruction.png'],
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
                           caption=CAPTIONS['sync_timeline.png'],
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
                           caption=CAPTIONS['alignment.png'],
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
                           caption=CAPTIONS['replicate_structure.png'],
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
                           caption=CAPTIONS['health.png'],
                           epilog=_coverage(tables), save=save, show=show)


# ------------------------------------------------------------------------------ the deliverable

def _embedded_figure(dataset: str, filename: str, heading: str) -> list:
    """One figure and its caption as markdown lines, or nothing if it has not been rendered.

    The caption comes from CAPTIONS, the same string the PNG carries, so the two cannot drift.
    An absent figure is skipped silently rather than left as a broken image link: --report-only
    is a supported way to run this, and it does not render anything.
    """
    figure = _plots_dir(dataset) / filename
    if not figure.exists():
        return []
    try:
        relative = figure.relative_to(paths.REPO_ROOT / 'results' / 'reports')
    except ValueError:
        relative = Path('../..') / figure.relative_to(paths.REPO_ROOT)
    return [f'## {heading}', '',
            f'![{heading}]({relative.as_posix()})', '',
            f'**Figure — {filename}.** {CAPTIONS.get(filename, "")}', '']


def write_report(tables: Dict[str, pd.DataFrame], dataset: str) -> Path:
    """The markdown report.

    Markdown rather than HTML because it is diffable: comparing this file before and after a
    pipeline change is exactly the review you want.

    It is NOT in version control, though — `results/` is gitignored, as is `plots/`, so both
    this file and the figures it embeds are build products. An earlier version of this
    docstring claimed the opposite. Keeping a copy across a pipeline change therefore means
    copying it somewhere else first.
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

    # data_state goes here, above the tables, because it answers "what state is this dataset
    # in" without reading anything else. The rest are gathered at the end: they answer
    # questions you already know to ask, and interleaving them with the tables would put a
    # figure between a claim and the numbers behind it.
    lines += _embedded_figure(dataset, 'data_state.png', 'At a glance')

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

            # A gap that repeats across every session is a protocol fact, not 42 separate
            # dropouts, and the per-trial list below buries it: at 25 rows it cannot even
            # show them all. Named here as one line each so the pattern is the finding.
            systematic = []
            for trial_name, group in coverage.groupby('trial'):
                sessions = group.subject.nunique()
                if sessions < 2:
                    continue
                never = (group.groupby('sensor')['present'].sum() == 0)
                for sensor in sorted(never[never].index):
                    reasons = group[group.sensor == sensor].status.value_counts()
                    systematic.append((trial_name, sensor, sessions, reasons.idxmax()))
            if systematic:
                # `neither` first would bury the finding: on the long walks it just means the
                # 7-sensor sessions have no High or Low placements, which is by design and
                # already explained in the status table above.
                systematic.sort(key=lambda row: (row[3] == 'neither', row[0], row[1]))
                lines += ['### Sensors missing from EVERY session of an activity',
                          '',
                          'Not dropouts, and not faults. A sensor absent from all sessions of '
                          'one activity is a property of the PROTOCOL, and the reason to '
                          'tabulate it is that the artifact cannot distinguish it from a '
                          'fault: either way the plate is simply not there.',
                          '',
                          'Both entries here are by design. IMoVE\'s treadmill activities '
                          'instrument the RIGHT LEG ONLY, so the left foot, shank and thigh '
                          'are absent from all 21 sessions that ran them — every left-leg '
                          'marker in a treadmill take reads 0.00% occupancy for the whole '
                          'take, against 99.99% for the same markers in the same session\'s '
                          'overground walking. The long-walk sessions carry 7 sensors rather '
                          'than 15, so they have no High or Low placements at all.',
                          '',
                          'Neither is a problem to fix. What matters downstream is that a '
                          'left-leg or a High/Low claim cannot be made from these trials, '
                          'and that any per-sensor rate computed over the whole dataset has '
                          'a denominator that varies by activity.',
                          '', '| activity | sensor | sessions | reason |',
                          '| --- | --- | --- | --- |']
                for trial_name, sensor, sessions, reason in systematic:
                    lines.append(f'| {trial_name} | `{sensor}` | {sessions} | `{reason}` |')
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

    # The rest of the figures, each with the caption its PNG carries. Gathered rather than
    # interleaved so a figure never lands between a claim and the table behind it.
    gallery = [('reconstruction.png', 'Figure — ground-truth quality'),
               ('sync_timeline.png', 'Figure — synchronization and timeline'),
               ('alignment.png', 'Figure — sensor-to-segment alignment'),
               ('replicate_structure.png', 'Figure — replicate structure'),
               ('health.png', 'Figure — worst trials and their components')]
    embedded = [line for filename, heading in gallery
                for line in _embedded_figure(dataset, filename, heading)]
    if embedded:
        lines += ['## Figures', '',
                  'Every figure below is also written to '
                  f'`plots/{PLOTS_SUBDIR}/{dataset}/` with its caption rendered onto the '
                  'image, so it stays readable when it travels without this report.', '']
        lines += embedded

    path = paths.ensure_parent(paths.REPO_ROOT / 'results' / 'reports' /
                               f'build_quality_{dataset}.md')
    path.write_text('\n'.join(lines))
    print(f"Wrote report to {path.relative_to(paths.REPO_ROOT)}")
    return path


FIGURES = (plot_data_state, plot_reconstruction, plot_sync_and_timeline, plot_alignment,
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
