"""
Figures for experiments/joint_dof.py: what a low-DOF joint model costs, and where it costs it.

Reads ONLY the tables that experiment wrote. Nothing here reloads a trial or refits a model, so
no number in a figure can disagree with the parquet beside it — the same tier discipline
plotting/build_quality.py follows, and for the same reason: a figure that recomputes is a second
implementation nobody diffs.

    python -m plotting.joint_dof --dataset alborno
    python -m plotting.joint_dof --dataset imove_biplane
    python -m plotting.joint_dof --dataset alborno --figures ladder coupling

Five figures, each answering one question the console report answers in prose:

  ladder      What does each degree of freedom buy? One panel per joint, the ladder as bars,
              cross-validated where available and marked when not.

  angle       WHERE does the hinge fail — everywhere, or only past mid-range? Error against the
              joint angle. This is the figure that decides whether a hinge constraint is safe
              for walking but not for a deep squat.

  coupling    The published Reuben knee coupling against the curve the joint actually traced,
              both as departure from the best geodesic through them, in degrees against the
              joint angle. The gap between the two is the part of the published shape that is
              wrong for these subjects rather than merely unmodelled.

  reference   Markers against biplane: the noise floor under every marker-based residual in the
              other four figures. Biplane dataset only.

  cohort      Per-subject spread of each model's error and of the mounting-invariant geometry,
              so a cohort median is never read without its dispersion.
"""
import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from plotting import utils as plot_utils
from experiments.joint_dof import DATASETS, dataset_dir, load_trial_table, n_trials

PLOTS_DIR = paths.plots_dir("joint_dof")

# One colour per rung, ordered as the ladder is: more freedom, cooler. Fixed here rather than
# taken from a palette so the same model is the same colour in every figure and across datasets.
MODEL_COLORS = {
    'weld': '#6b6b6b',
    'hinge': '#b4432f',
    'universal': '#4c9f70',
    'spherical': '#3f6fa8',
    'knee_coupling': '#8a4fa8',
}
MODEL_LABELS = {
    'weld': '0-DOF weld',
    'hinge': '1-DOF hinge',
    'universal': '2-DOF universal',
    'spherical': '3-DOF spherical',
    'knee_coupling': '1-DOF Reuben coupling',
}

# The weld is drawn only where it is the POINT — it is the denominator, and on a joint with
# 40 deg of range it is ten times every other bar and flattens them all into the axis.
LADDER_MODELS = ('hinge', 'knee_coupling', 'universal')

SOURCE_STYLE = {'markers': dict(marker='o', linestyle='-'),
                'vicon': dict(marker='o', linestyle='-'),
                'biplane': dict(marker='s', linestyle='--')}

FIGURES = ('ladder', 'angle', 'coupling', 'reference', 'cohort')


def _metric(fits: pd.DataFrame) -> str:
    """Cross-validated error where it exists, in-sample otherwise — and say which.

    Never silently: an in-sample ladder flatters the coupling by its extra structure
    parameters, which is the one comparison in these figures that a reader could be misled by.
    """
    if 'cv_rms_deg' in fits.columns and fits['cv_rms_deg'].notna().any():
        return 'cv_rms_deg'
    return 'rms_deg'


def _metric_note(metric: str) -> str:
    return ('Cross-validated (out-of-fold) RMS.' if metric.startswith('cv_') else
            'IN-SAMPLE RMS — cross-validation did not run, so the coupling is flattered by its '
            'extra structure parameters and its margin over the hinge should not be trusted.')


def _joint_order(fits: pd.DataFrame) -> List[str]:
    """Proximal to distal, then anything unrecognised, so every figure sorts the same way."""
    order = ['Lumbar', 'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'R_Ankle', 'L_Ankle']
    present = list(dict.fromkeys(fits['joint']))
    return [j for j in order if j in present] + sorted(set(present) - set(order))


# ==============================================================================
# 1. The ladder
# ==============================================================================

def plot_ladder(dataset: str, fits: pd.DataFrame, save: bool = True, show: bool = False) -> None:
    """Median error per model per joint, with the per-trial spread behind it."""
    metric = _metric(fits)
    joints = _joint_order(fits)
    sources = sorted(fits['source'].unique())
    fig, axes = plt.subplots(1, len(sources), figsize=(1.6 * len(joints) * len(sources) + 3, 5.6),
                             squeeze=False, sharey=True)

    models = [m for m in LADDER_MODELS if m in set(fits['model'])]
    width = 0.8 / len(models)

    for column, source in enumerate(sources):
        ax = axes[0][column]
        subset = fits[fits['source'] == source]
        x = np.arange(len(joints))
        for index, model in enumerate(models):
            offset = (index - (len(models) - 1) / 2) * width
            values, spreads = [], []
            for joint in joints:
                rows = subset[(subset['joint'] == joint) & (subset['model'] == model)][metric]
                rows = rows.dropna()
                values.append(rows.median() if len(rows) else np.nan)
                # Inter-quartile range across trials, not a standard error: these are trials of
                # the same subjects, so they are not independent and an error bar implying they
                # were would be a claim this figure has no basis for.
                spreads.append([rows.median() - rows.quantile(0.25) if len(rows) else 0,
                                rows.quantile(0.75) - rows.median() if len(rows) else 0])
            ax.bar(x + offset, values, width=width, color=MODEL_COLORS[model],
                   label=MODEL_LABELS[model],
                   yerr=np.array(spreads).T, capsize=2,
                   error_kw=dict(elinewidth=1.0, ecolor='#333333'))
        ax.set_xticks(x)
        ax.set_xticklabels(joints, rotation=30, ha='right')
        ax.set_title(f"reference: {source}" if len(sources) > 1 else '')
        if column == 0:
            ax.set_ylabel('geodesic error (deg)')
    axes[0][-1].legend(frameon=False, fontsize=10)

    plot_utils.finalize_and_save_plot(
        fig, f"What each degree of freedom buys — {dataset}",
        f"joint_dof_ladder_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"bars = median over trials, whiskers = IQR; n={n_trials(fits)} trials",
        caption=(
            f"Geodesic error of each joint model, fitted to the reference relative rotation with "
            f"no filter and no IMU orientation estimate anywhere in the path. {_metric_note(metric)} "
            f"The models NEST — a hinge is the coupling with no curvature, the coupling is a "
            f"joint with its second angle frozen — so the bars are monotone by construction and "
            f"universal joint with one axis frozen — so each drop is what the extra freedom is "
            f"worth. The hinge and the coupling carry the "
            f"SAME one angle per sample and differ only in whether that angle traces a geodesic, "
            f"so their gap is curvature alone: a coupling linear in the joint angle is a tilted "
            f"hinge and the hinge's axis is already free. The Reuben coupling is that same "
            f"one-angle joint with the off-axis channels set to a PUBLISHED curve instead of to "
            f"zero, so it is drawn between them and it is only fitted at the knees. The 0-DOF weld and the 3-DOF spherical "
            f"joint are omitted — the first is ten times the others and flattens the panel, the "
            f"second is identically zero because a ball joint constrains no relative orientation. "
            f"NOT shown here: whether the joint moved at all. A joint with little range is fitted "
            f"well by everything; see the error-as-a-fraction-of-weld column in the console "
            f"report."))


# ==============================================================================
# 2. Error against joint angle
# ==============================================================================

def plot_angle(dataset: str, curve: pd.DataFrame, save: bool = True,
               show: bool = False) -> None:
    """Where in the range of motion each model fails."""
    if curve.empty:
        print("  angle: no error_curve tables on disk.")
        return
    joints = _joint_order(curve)
    sources = sorted(curve['source'].unique())
    columns = min(4, len(joints))
    rows = int(np.ceil(len(joints) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows),
                             squeeze=False, sharey=True)

    for index, joint in enumerate(joints):
        ax = axes[index // columns][index % columns]
        for source in sources:
            subset = curve[(curve['joint'] == joint) & (curve['source'] == source)]
            if subset.empty:
                continue
            # Pooled across trials by binning on the bin centre, which is comparable across
            # trials because every trial's bins span its own 0.5-99.5 percentile of the SAME
            # quantity: the fitted 1-DOF angle.
            for model in [m for m in LADDER_MODELS if f'{m}_rms_deg' in subset.columns]:
                grouped = subset.groupby(pd.cut(subset['angle_mid_deg'], 18),
                                         observed=True)[f'{model}_rms_deg'].median()
                centres = [interval.mid for interval in grouped.index]
                ax.plot(centres, grouped.to_numpy(), color=MODEL_COLORS[model],
                        label=f"{MODEL_LABELS[model]} ({source})" if index == 0 else None,
                        **SOURCE_STYLE.get(source, SOURCE_STYLE['markers']),
                        markersize=3, linewidth=1.6)
        ax.set_title(joint)
        if index % columns == 0:
            ax.set_ylabel('RMS error (deg)')
        if index // columns == rows - 1:
            ax.set_xlabel('joint angle (deg)')
    for spare in range(len(joints), rows * columns):
        axes[spare // columns][spare % columns].axis('off')
    axes[0][0].legend(frameon=False, fontsize=8)

    plot_utils.finalize_and_save_plot(
        fig, f"Where in the range of motion each model fails — {dataset}",
        f"joint_dof_angle_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog="median over trials, binned on the fitted 1-DOF angle",
        caption=(
            "Model error against the joint's own 1-DOF angle, so a model that is fine through "
            "mid-range and fails at end range is distinguishable from one that is uniformly "
            "mediocre — a distinction the RMS in the ladder figure cannot make, and the one that "
            "decides whether a hinge constraint is safe for level walking but not for stair "
            "descent or a deep squat. Every model is binned on the HINGE angle so the bins mean "
            "the same thing across a row. The zero of the angle is the fitted neutral pose, "
            "which is per trial and per subject, so the horizontal axis is comparable in WIDTH "
            "across panels but its origin is not anatomical. Bins holding fewer than five "
            "samples are dropped, so the curves stop short of the extreme range each joint "
            "visited only briefly."))


# ==============================================================================
# 3. The fitted coupling
# ==============================================================================

def plot_coupling(dataset: str, shape: pd.DataFrame, save: bool = True,
                  show: bool = False) -> None:
    """The published Reuben coupling against the curve the joint actually traced."""
    if shape.empty:
        print("  coupling: no coupling_shape tables on disk.")
        return
    joints = _joint_order(shape)
    sources = sorted(shape['source'].unique())
    columns = min(4, len(joints))
    rows = int(np.ceil(len(joints) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)

    drawn = set()
    for index, joint in enumerate(joints):
        ax = axes[index // columns][index % columns]
        for source in sources:
            for model in ('knee_coupling',):
                subset = shape[(shape['joint'] == joint) & (shape['source'] == source)
                               & (shape['model'] == model)]
                if subset.empty:
                    continue
                # One faint line per trial behind the median, because the SPREAD is the finding
                # on the marker datasets: a tight bundle means the curve reproduces between
                # trials and a fan means it is chasing whichever motion each trial contained.
                # Per-trial faints show how much the FITTING of the curve's placement moved
                # time by construction, so nineteen copies of it would be nineteen copies.
                if model == 'knee_coupling':
                    for _, trial_curve in subset.groupby(['subject', 'trial'], observed=True):
                        trial_curve = trial_curve.sort_values('angle_deg')
                        ax.plot(trial_curve['angle_deg'],
                                trial_curve['coupling_magnitude_deg'],
                                color=MODEL_COLORS['knee_coupling'], alpha=0.15, linewidth=0.7)
                grouped = subset.groupby(pd.cut(subset['angle_deg'], 24),
                                         observed=True)['coupling_magnitude_deg'].median()
                label = f"{MODEL_LABELS[model]}" + (f" ({source})" if len(sources) > 1 else '')
                ax.plot([i.mid for i in grouped.index], grouped.to_numpy(),
                        color=MODEL_COLORS[model], linewidth=2.4,
                        linestyle='--' if source == 'biplane' else '-',
                        label=None if label in drawn else label)
                drawn.add(label)
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_title(joint)
        if index % columns == 0:
            ax.set_ylabel('off-axis rotation (deg)')
        if index // columns == rows - 1:
            ax.set_xlabel('joint angle (deg)')
    for spare in range(len(joints), rows * columns):
        axes[spare // columns][spare % columns].axis('off')
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=9, loc='upper right')

    plot_utils.finalize_and_save_plot(
        fig, f"The published knee coupling against the curve the joint traced — {dataset}",
        f"joint_dof_coupling_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog="faint = one trial, bold = median; off-axis departure of the published curve",
        caption=(
            "Off-axis rotation against joint angle: the form a published coupling is quoted in. "
            "Purple is the Reuben coupling as IMoveLab enforces it, orange is the curve this "
            "joint actually traced, and the gap between them is the part of the published shape "
            "that is wrong for these subjects rather than merely unmodelled. Both are measured "
            "as the departure from the best GEODESIC through the curve, not from the model's "
            "own neutral pose, because only the former is a property of the joint — the "
            "optimizer is free to move the neutral pose, which slides curvature into and out of "
            "the coefficients at will. It is also the right comparison: the LINEAR part of any "
            "coupling is a hinge about a tilted axis, and a free-axis hinge already gets that "
            "for nothing, so only what is plotted here was ever worth extra structure. Magnitude "
            "rather than the two channels separately, because their split is arbitrary up to a "
            "rotation of the plane perpendicular to the axis and no care can fix that without "
            "an anatomical frame. On the marker datasets the frames are sensor plates "
            "re-strapped per subject, so the spread between faint lines is mounting as much as "
            "anatomy; on the biplane source the frames are bone-fixed and the spread is real."))


# ==============================================================================
# 4. Markers against biplane
# ==============================================================================

def plot_reference(dataset: str, agreement: pd.DataFrame, fits: pd.DataFrame,
                   save: bool = True, show: bool = False) -> None:
    """The reference-disagreement floor, against the model residuals it has to be read under."""
    if agreement.empty:
        print("  reference: this dataset has one reference system; nothing to compare.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax = axes[0]
    joints = _joint_order(agreement)
    data = [agreement.loc[agreement['joint'] == j, 'agreement_rms_deg'].dropna() for j in joints]
    ax.boxplot(data, tick_labels=joints, showfliers=False,
               medianprops=dict(color='black', linewidth=2))
    for index, values in enumerate(data):
        ax.plot(np.full(len(values), index + 1) + np.random.default_rng(0).uniform(
            -0.12, 0.12, len(values)), values, 'o', markersize=3, alpha=0.4, color='#3f6fa8')
    ax.set_ylabel('marker-vs-biplane disagreement (deg RMS)')
    ax.set_title('The floor: how far apart the two references are')

    ax = axes[1]
    floor = float(agreement['agreement_rms_deg'].median())
    metric = _metric(fits)
    sources = sorted(fits['source'].unique())
    models = [m for m in LADDER_MODELS if m in set(fits['model'])]
    width = 0.8 / max(len(sources), 1)
    x = np.arange(len(models))
    for index, source in enumerate(sources):
        offset = (index - (len(sources) - 1) / 2) * width
        values = [fits.loc[(fits['source'] == source) & (fits['model'] == m), metric].median()
                  for m in models]
        ax.bar(x + offset, values, width=width, label=source,
               color=['#b0b0b0', '#3f6fa8'][index % 2])
    ax.axhline(floor, color='#c0243a', linestyle='--', linewidth=1.6,
               label=f'reference disagreement ({floor:.2f} deg)')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=20, ha='right')
    ax.set_ylabel('model error (deg RMS)')
    ax.set_title('Model residuals against that floor')
    ax.legend(frameon=False, fontsize=9)

    plot_utils.finalize_and_save_plot(
        fig, f"Markers versus biplane — {dataset}",
        f"joint_dof_reference_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"n={len(agreement)} trial-joint comparisons",
        caption=(
            "Left: the same knee over the same frames, measured by skin-mounted Vicon marker "
            "clusters and by biplane fluoroscopy solving the bone poses directly. The two frames "
            "differ by a constant — a cluster's frame is wherever the template landed on the "
            "skin, a bone frame is anatomy — so that constant is fitted away and what is plotted "
            "is what survives it: soft-tissue artifact plus marker error plus whatever the "
            "fluoroscopy got wrong. Right: the same quantity as a line under the model residuals. "
            "A marker-based residual BELOW this line is not measuring the joint, it is measuring "
            "the markers, and the interesting comparison is between the two sources' bars at the "
            "same rung: where they agree, the model's limitation is real kinematics; where the "
            "marker bar is higher, part of it is the skin. This floor exists only on the biplane "
            "half, but it is the right scale for reading the marker-only datasets too."))


# ==============================================================================
# 5. Cohort spread
# ==============================================================================

def plot_cohort(dataset: str, fits: pd.DataFrame, pooled: pd.DataFrame,
                save: bool = True, show: bool = False) -> None:
    """Per-subject spread, so a cohort median is never read without its dispersion."""
    metric = _metric(fits)
    joints = _joint_order(fits)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    # Offsets derived from how many models are actually PRESENT, not from their index in the
    # full list: the Reuben coupling exists only at the knee, so a fixed index leaves a gap on
    # every other joint and overlaps the neighbouring group's boxes.
    present = [m for m in LADDER_MODELS if m in set(fits['model'])]
    span = 0.8 / max(len(present), 1)
    for index, model in enumerate(present):
        per_subject = (fits[fits['model'] == model]
                       .groupby(['joint', 'subject'], observed=True)[metric].median()
                       .reset_index())
        data = [per_subject.loc[per_subject['joint'] == j, metric].dropna() for j in joints]
        positions = np.arange(len(joints)) + (index - (len(present) - 1) / 2) * span
        ax.boxplot(data, positions=positions, widths=0.85 * span, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=MODEL_COLORS[model], alpha=0.65),
                   medianprops=dict(color='black', linewidth=1.6))
    ax.set_xticks(np.arange(len(joints)))
    ax.set_xticklabels(joints, rotation=30, ha='right')
    ax.set_ylabel('error (deg)')
    ax.set_title('Between-subject spread of each model')

    ax = axes[1]
    universal = fits[fits['model'] == 'universal']
    if 'carrying_angle_deg' in universal.columns:
        per_subject = universal.groupby(['joint', 'subject'],
                                        observed=True)['carrying_angle_deg'].median().reset_index()
        ax.boxplot([per_subject.loc[per_subject['joint'] == j, 'carrying_angle_deg'].dropna()
                    for j in joints], tick_labels=joints, showfliers=False,
                   medianprops=dict(color='black', linewidth=2))
        ax.axhline(30.0, color='black', linestyle=':', linewidth=1.2)
        ax.text(len(joints) + 0.4, 30.0, 'not identified below', va='bottom', ha='right',
                fontsize=9)
    ax.set_xticklabels(joints, rotation=30, ha='right')
    ax.set_ylabel('carrying angle (deg)')
    ax.set_title('2-DOF geometry, the one poolable quantity')

    ax = axes[2]
    if not pooled.empty:
        merged = []
        for joint in joints:
            per_trial = fits[(fits['joint'] == joint) & (fits['model'] == 'hinge')]['rms_deg']
            per_session = pooled[(pooled['joint'] == joint)
                                 & (pooled['model'] == 'hinge')]['rms_deg']
            if per_trial.empty or per_session.empty:
                continue
            merged.append((joint, per_trial.median(), per_session.median()))
        if merged:
            labels, trial_values, session_values = zip(*merged)
            x = np.arange(len(labels))
            ax.bar(x - 0.2, trial_values, 0.4, label='per trial', color='#b0b0b0')
            ax.bar(x + 0.2, session_values, 0.4, label='pooled per session',
                   color=MODEL_COLORS['hinge'])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.legend(frameon=False, fontsize=9)
    ax.set_ylabel('hinge error (deg)')
    ax.set_title('Does one model serve a whole session?')

    plot_utils.finalize_and_save_plot(
        fig, f"Between-subject and between-trial consistency — {dataset}",
        f"joint_dof_cohort_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"n={fits['subject'].nunique()} subjects, {n_trials(fits)} trials",
        caption=(
            "Left: each model's error per subject, so a cohort median is never read without the "
            "spread behind it. Middle: the carrying angle — the constant angle between the two "
            "fitted axes of the universal joint — which is the ONE geometric quantity these "
            "datasets can pool across subjects. Remounting a sensor by a constant rotation sends "
            "both the axis and the neutral pose with it and leaves the angle between them "
            "unchanged, whereas the axis VECTORS live in each subject's own re-strapped plate "
            "frame and are not comparable at all; they appear in no figure here for that reason. "
            "Below about 30 deg the two axes are near parallel, the two joint angles have a "
            "near-null direction, and the carrying angle is not identified — the RESIDUAL is "
            "still a real achieved number there, but the geometry is not. Right: a hinge fitted "
            "once over a whole session against the per-trial fits it pools. A pooled bar far "
            "above the per-trial one means the structure moved between trials; on the biplane "
            "half it is the per-trial bar to distrust instead, because half a second of one hop "
            "does not identify an axis however small its residual."))


# ==============================================================================
# CLI
# ==============================================================================

def load_tables(dataset: str) -> Dict[str, pd.DataFrame]:
    tables = {name: load_trial_table(dataset, name)
              for name in ('joint_fits', 'error_curve', 'coupling_shape', 'reference_agreement')}
    pooled_path = dataset_dir(dataset) / 'pooled_fits.parquet'
    tables['pooled'] = (pd.read_parquet(pooled_path, engine='pyarrow') if pooled_path.exists()
                        else pd.DataFrame())
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--figures', nargs='+', choices=FIGURES, default=list(FIGURES),
                        metavar='FIGURE')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    tables = load_tables(args.dataset)
    fits = tables['joint_fits']
    if fits.empty:
        print(f"No joint_dof tables under {dataset_dir(args.dataset)}. "
              f"Run: python -m experiments.joint_dof --dataset {args.dataset}")
        return 1
    save, show = not args.no_save, args.show

    if 'ladder' in args.figures:
        plot_ladder(args.dataset, fits, save, show)
    if 'angle' in args.figures:
        plot_angle(args.dataset, tables['error_curve'], save, show)
    if 'coupling' in args.figures:
        plot_coupling(args.dataset, tables['coupling_shape'], save, show)
    if 'reference' in args.figures:
        plot_reference(args.dataset, tables['reference_agreement'], fits, save, show)
    if 'cohort' in args.figures:
        plot_cohort(args.dataset, fits, tables['pooled'], save, show)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
