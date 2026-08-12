"""
Paper figure 2 — the two disturbances the EKF cannot reject.

Panel A pools every subject/trial/joint: joint-angle error for the EKF, its two
oracle variants, and MAJIC. Panels B and C zoom into one trial to show *where*
the EKF's error comes from, each holding one disturbance fixed while the other
varies:

  Panel B — field constant, motion varies. Opens inside a stationary bout (the
            accelerometer reads gravity to within ~1 deg) and runs into the
            movement that follows. The acc oracle tracks; the real EKF doesn't.
  Panel C — motion constant, field varies. Sits entirely inside steady
            locomotion, spanning a walk from clean field into distorted field.
            The mag oracle tracks; the real EKF doesn't.

Under each error trace is the disturbance that drives it, defined as the angle
between the real sensor reading and the oracle reading that replaced it — so the
bottom row is literally "how wrong was the sensor the oracle line fixed", in the
same units (degrees) as the error above it.

Reads results/joint_angles/**.parquet and results/statistics/ekf_oracle_comparison_statistics.parquet,
both produced by experiments/ekf_oracle_comparison.py.
"""
import os
os.environ.setdefault("DISABLE_TQDM", "True")

import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.transform import Rotation

import paths
from experiments.experiment_utils import (
    JOINTS, load_raw_data, load_joint_angles, load_statistics,
    _compute_expected_mag_field, _compute_perfect_segment_acc, _compute_perfect_mag,
)
from experiments.drift_observability import detect_quiet_sitting_segments
from plotting.utils import (
    DEFAULT_PALETTE, draw_distribution, test_panels, significance_report,
    _emit_significance_report,
)

PLOTS_DIR = paths.plots_dir("ekf_oracle_comparison")

# --- What's compared ---------------------------------------------------------

BASELINE_METHOD = 'ekf_rescaled'
ACC_ORACLE_METHOD = 'ekf_rescaled_perfect_acc'
MAG_ORACLE_METHOD = 'ekf_rescaled_perfect_mag'
BOTH_ORACLE_METHOD = 'ekf_rescaled_perfect_acc_perfect_mag'

PANEL_A_METHODS = [BASELINE_METHOD, MAG_ORACLE_METHOD, ACC_ORACLE_METHOD,
                   BOTH_ORACLE_METHOD, 'mag_on']

# The two zoom panels are a *nested* ablation, not two parallel ones, because the
# disturbances are not symmetric on this dataset: the accelerometer disturbance
# dominates so completely that an oracle magnetometer on its own changes nothing
# (see panel A). So panel B asks "does fixing the accelerometer help?" and panel C
# asks "once the accelerometer is fixed, does fixing the magnetometer help too?".
# Pairing the mag panel against the raw EKF instead would show a flat null and
# say nothing about whether magnetic distortion costs anything.
PANEL_B_METHODS = [BASELINE_METHOD, ACC_ORACLE_METHOD, 'mag_on']
PANEL_C_METHODS = [ACC_ORACLE_METHOD, BOTH_ORACLE_METHOD, 'mag_on']
# Carried through both zoom panels as a reference rather than as half of an
# ablation pair, so it's drawn dashed and lighter than the pair being compared.
REFERENCE_METHOD = 'mag_on'
ZOOM_METHODS = sorted(set(PANEL_B_METHODS) | set(PANEL_C_METHODS))

METHOD_LABELS = {
    BASELINE_METHOD: 'EKF (real/real)',
    MAG_ORACLE_METHOD: 'EKF (oracle mag)',
    ACC_ORACLE_METHOD: 'EKF (oracle acc)',
    BOTH_ORACLE_METHOD: 'EKF (oracle acc + mag)',
    'mag_on': 'MAJIC (real/real)',
}

RENAME_JOINTS = {'R_Hip': 'Hip', 'L_Hip': 'Hip', 'R_Knee': 'Knee', 'L_Knee': 'Knee',
                 'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle'}

METRIC, METRIC_UNITS, AXIS_TO_PLOT = 'rmse', 'deg', 'MAG'

# --- Display trial -----------------------------------------------------------
# complexTasks is the only activity with stationary bouts (walking has none), and
# it carries the wider magnetic excursion, so both zoom panels come from it.
DISPLAY_ACTIVITY = 'complexTasks'
# A distal joint: magnetic distortion in this dataset is floor-driven and an order
# of magnitude worse at the foot than at the pelvis, and the ankle's parent/child
# pair (tibia, calcaneus) shows that stratification in the bottom row for free.
DISPLAY_JOINT = 'R_Ankle'

# --- Window selection --------------------------------------------------------
# _run_relative_filter seeds the filter with the ground-truth orientation at t=0,
# so error is zero by construction at trial start and any early window looks
# misleadingly good. Skip the run-up.
MIN_WINDOW_START_SEC = 60.0

ACC_PRE_SEC = 10.0            # stationary run-in shown before the movement onset
ACC_POST_SEC = 25.0           # movement shown after it
ACC_MIN_STILL_SEC = 8.0       # a bout shorter than this can't anchor the panel
ACC_ONSET_DEG = 10.0          # smoothed acc deviation that counts as "moving again"
ACC_MAX_ONSET_LAG_SEC = 40.0  # give up looking for movement this long after a bout

MAG_WINDOW_SEC = 30.0
MAG_STEP_SEC = 2.0
MAG_MAX_ACC_IQR_DEG = 12.0  # "steady motion": the acc disturbance mustn't also swing
MAG_MIN_TREND_DEG = 10.0    # the field must actually get worse across the window

SMOOTH_SEC = 1.0             # bottom-row heavy line, and the movement-onset gate
MAG_SMOOTH_SEC = 2.0         # bottom-row heavy line for the stride-modulated mag signal
MAG_TREND_SMOOTH_SEC = 8.0   # selection only: strips stride peaks to leave positional drift

DISTURBANCE_COLORS = {'parent': '#4C566A', 'child': '#BF616A'}

# ==============================================================================
# Disturbance signals
# ==============================================================================

def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample angle (deg) between two stacks of 3-vectors.

    A dropped sample reads as an exact zero vector in this dataset; clamping the
    norm keeps that from turning into a NaN that would silently gap the trace.
    """
    unit = lambda v: v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=1), -1.0, 1.0)))

def _smooth(x: np.ndarray, seconds: float, fs: float) -> np.ndarray:
    n = max(int(round(seconds * fs)), 1)
    return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()

def sensor_disturbances(plates: Dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-sensor deviation of the real reading from the oracle that replaced it.

    'acc' is the angle between the measured accelerometer vector and gravity
    rotated into the segment's true orientation — i.e. exactly what
    ekf_perfect_acc substitutes. 'mag' is the same for the magnetometer against
    the trial's median world field. Both in degrees, directly comparable to the
    joint-angle error they drive.
    """
    expected_mag = _compute_expected_mag_field(list(plates.values()))
    return {
        name: {
            'acc': _angle_between(plate.imu_trace.acc, _compute_perfect_segment_acc(plate)),
            'mag': _angle_between(plate.imu_trace.mag, _compute_perfect_mag(plate, expected_mag)),
        }
        for name, plate in plates.items()
    }

# ==============================================================================
# Per-timestep joint-angle error
# ==============================================================================

def joint_error_trace(subject: str, activity: str, method: str,
                      joint: str) -> Optional[Tuple[np.ndarray, Rotation]]:
    """Per-sample joint-angle error rotation of `method` against marker.

    Its magnitude is the same quantity compute_error_stats aggregates into the
    RMSE of panel A, so the zoom panels and the summary panel measure the same
    thing. The rotation itself is kept rather than just the angle because the zoom
    panels need error *growth* relative to a window start (see drift_since).
    """
    est = load_joint_angles(subject, activity, method)
    truth = load_joint_angles(subject, activity, 'marker')
    if est is None or truth is None:
        return None

    est = est[est['joint_name'] == joint]
    truth = truth[truth['joint_name'] == joint]
    merged = pd.merge(est, truth, on=['timestamp', 'joint_name'],
                      suffixes=('_est', '_true')).sort_values('timestamp')
    if merged.empty:
        return None

    r_est = Rotation.from_rotvec(merged[['rx_est', 'ry_est', 'rz_est']].to_numpy())
    r_true = Rotation.from_rotvec(merged[['rx_true', 'ry_true', 'rz_true']].to_numpy())
    return merged['timestamp'].to_numpy(), r_est * r_true.inv()

def drift_since(error_rot: Rotation, i0: int) -> np.ndarray:
    """Error accumulated since sample i0, in degrees.

    Absolute error is path-dependent: by the time a 30 s window opens, the EKF may
    already be 150 deg off from something that happened minutes earlier, and the
    trace shows a level rather than a response. Re-referencing to the window start
    — the same operation drift_observability.segment_drift performs — isolates the
    error this window is responsible for, which is what the panels are claiming.

    Equal to the angle between the error rotation at t0 and at t, since
    (R_est0^T R_est)^T (R_true0^T R_true) is a conjugation of E(t0) E(t)^T and
    conjugation preserves rotation angle.
    """
    return np.degrees((error_rot[i0] * error_rot.inv()).magnitude())

def load_zoom_errors(subject: str, activity: str,
                     joint: str) -> Dict[str, Tuple[np.ndarray, Rotation]]:
    traces = {m: joint_error_trace(subject, activity, m, joint) for m in ZOOM_METHODS}
    return {m: t for m, t in traces.items() if t is not None}

# ==============================================================================
# Window selection
# ==============================================================================

def _joint_sensors(joint: str) -> Tuple[str, str]:
    return JOINTS[joint]

def _gap_at_end(errors: Dict, worse: str, better: str, i0: int, i1: int,
                fs: float) -> Optional[float]:
    """How much more error `worse` accumulated than `better` over [i0, i1].

    This is the attribution signal, and it's what both selectors maximize. Scoring
    on the disturbance alone picks windows where the sensor misbehaves but the
    filter shrugs it off; scoring on the oracle gap picks windows where the
    disturbance demonstrably caused the error, which is the claim being made.
    Averaged over the last few seconds rather than read off the final sample, so a
    single spike can't win.
    """
    if worse not in errors or better not in errors:
        return None
    tail = slice(max(i1 - int(3.0 * fs), i0 + 1), i1)
    if tail.stop <= tail.start:
        return None
    return float(drift_since(errors[worse][1], i0)[tail].mean()
                 - drift_since(errors[better][1], i0)[tail].mean())

def select_acc_window(plates: Dict, disturb: Dict, errors: Dict, joint: str) -> Optional[Dict]:
    """Best stationary-bout -> movement transition for the acc panel.

    The window is anchored on the *movement onset*, not the end of the detected
    bout: the pelvis can leave the bout a good ten seconds before the limbs
    actually start accelerating, and anchoring on the bout end buries the onset
    mid-panel. Among qualifying bouts, the one whose following movement opens the
    widest acc-oracle gap wins.
    """
    parent, child = _joint_sensors(joint)
    if 'pelvis_imu' not in plates or parent not in plates or child not in plates:
        return None

    timestamps = plates[parent].imu_trace.timestamps
    fs = 1.0 / np.mean(np.diff(timestamps))
    pair_acc = _smooth(np.maximum(disturb[parent]['acc'], disturb[child]['acc']), SMOOTH_SEC, fs)

    bouts = detect_quiet_sitting_segments(plates['pelvis_imu'], min_duration_sec=ACC_MIN_STILL_SEC)
    best = None
    for start, end in bouts:
        end = min(end, len(timestamps) - 1)
        if timestamps[start] < MIN_WINDOW_START_SEC or end <= start:
            continue

        search = slice(end, min(end + int(ACC_MAX_ONSET_LAG_SEC * fs), len(timestamps)))
        moving = np.flatnonzero(pair_acc[search] > ACC_ONSET_DEG)
        if moving.size == 0:
            continue
        onset = end + int(moving[0])

        i0 = max(onset - int(ACC_PRE_SEC * fs), start)
        i1 = min(onset + int(ACC_POST_SEC * fs), len(timestamps) - 1)
        score = _gap_at_end(errors, BASELINE_METHOD, ACC_ORACLE_METHOD, i0, i1, fs)
        if score is None:
            continue

        if best is None or score > best['score']:
            best = {
                'score': score, 'i0': i0, 'i1': i1,
                't0': timestamps[i0], 't1': timestamps[i1],
                'onset_t': timestamps[onset],
                'still_deg': float(pair_acc[i0:onset].mean()),
                'moving_deg': float(pair_acc[onset:i1].mean()),
            }
    return best

def select_mag_window(plates: Dict, disturb: Dict, errors: Dict, joint: str,
                      exclude: Optional[Tuple[float, float]] = None) -> Optional[Dict]:
    """Largest magnetically-attributable error growth at steady motion.

    Slides a fixed window over everything outside the stationary bouts and keeps
    only those where the acc disturbance is roughly stationary, so the panel isn't
    confounded by the *other* failure mode. Among survivors, requires the field
    deviation to actually trend upward across the window — measured as a
    first-third to last-third shift of the heavily-smoothed signal, not a max-minus-min,
    which at the foot is dominated by per-stride peaks rather than by the sustained
    positional drift the panel is about — and then maximizes the magnetic gap.

    The gap is measured between the two acc-oracle variants, not against the raw
    EKF: with a real accelerometer the filter is already lost, and any window would
    score high for reasons that have nothing to do with the magnetometer.

    `exclude` blocks a time range (the acc panel's window) so the two zoom panels
    can't land on top of each other and show the reader the same seconds twice.
    """
    parent, child = _joint_sensors(joint)
    if 'pelvis_imu' not in plates or parent not in plates or child not in plates:
        return None

    timestamps = plates[parent].imu_trace.timestamps
    fs = 1.0 / np.mean(np.diff(timestamps))
    pair_acc = _smooth(np.maximum(disturb[parent]['acc'], disturb[child]['acc']), SMOOTH_SEC, fs)
    child_mag = _smooth(disturb[child]['mag'], MAG_TREND_SMOOTH_SEC, fs)

    still = np.zeros(len(timestamps), dtype=bool)
    for start, end in detect_quiet_sitting_segments(plates['pelvis_imu'], min_duration_sec=3.0):
        still[start:min(end, len(still))] = True

    width, step, third = int(MAG_WINDOW_SEC * fs), int(MAG_STEP_SEC * fs), int(MAG_WINDOW_SEC * fs / 3)
    best = None
    for start in range(int(MIN_WINDOW_START_SEC * fs), len(timestamps) - width, step):
        stop = start + width
        window = slice(start, stop)
        if still[window].any():
            continue
        if exclude is not None and timestamps[start] < exclude[1] and timestamps[stop] > exclude[0]:
            continue
        acc_iqr = float(np.subtract(*np.percentile(pair_acc[window], [75, 25])))
        if acc_iqr > MAG_MAX_ACC_IQR_DEG:
            continue

        low = float(child_mag[start:start + third].mean())
        high = float(child_mag[stop - third:stop].mean())
        if high - low < MAG_MIN_TREND_DEG:
            continue

        score = _gap_at_end(errors, ACC_ORACLE_METHOD, BOTH_ORACLE_METHOD,
                            start, stop, fs)
        if score is None:
            continue
        if best is None or score > best['score']:
            best = {'score': score, 'i0': start, 'i1': stop,
                    't0': timestamps[start], 't1': timestamps[stop],
                    'low_deg': low, 'high_deg': high, 'acc_iqr_deg': acc_iqr}
    return best

def rank_subjects_by_oracle_gap(stats_df: pd.DataFrame, activity: str) -> List[str]:
    """Subjects ordered by how close their EKF-minus-oracle gap is to the median.

    Picking the display subject by *median* gap rather than by the most dramatic
    window is the cheap defence against a cherry-picking objection: the reader gets
    a typical subject, not the best one.
    """
    metric_col = f"{METRIC}_{METRIC_UNITS}"
    df = stats_df[(stats_df['axis'] == AXIS_TO_PLOT) & (stats_df['trial_type'] == activity)]
    pivot = df.pivot_table(index='subject', columns='method', values=metric_col, aggfunc='mean')
    oracles = [m for m in (ACC_ORACLE_METHOD, MAG_ORACLE_METHOD) if m in pivot.columns]
    if BASELINE_METHOD not in pivot.columns or not oracles:
        return []

    gap = (pivot[BASELINE_METHOD].to_frame().values - pivot[oracles].values).mean(axis=1)
    gap = pd.Series(gap, index=pivot.index).dropna()
    if gap.empty:
        return []
    return list((gap - gap.median()).abs().sort_values().index)

def select_display_subject(stats_df: pd.DataFrame, activity: str, joint: str,
                           forced: Optional[str] = None) -> Optional[Dict]:
    """Walks subjects from most-typical outward, returning the first that can
    actually supply both windows. Subjects with no stationary bout (01-03 in this
    dataset) fail the acc selection and are skipped automatically."""
    if forced:
        candidates = [f"Subject{forced}"]
    else:
        candidates = rank_subjects_by_oracle_gap(stats_df, activity)
        if not candidates:
            print("Could not rank subjects by oracle gap — is the statistics parquet present?")
            return None

    for name in candidates:
        subject = name.replace('Subject', '')
        try:
            plates = load_raw_data(subject, activity)
        except Exception as exc:
            print(f"  {name}: raw data unavailable ({exc})")
            continue

        errors = load_zoom_errors(subject, activity, joint)
        missing = [m for m in ZOOM_METHODS if m not in errors]
        if missing:
            print(f"  {name}: joint angles missing for {missing} — skipping")
            continue

        disturb = sensor_disturbances(plates)
        acc_win = select_acc_window(plates, disturb, errors, joint)
        if acc_win is None:
            print(f"  {name}: no stationary bout followed by movement — skipping")
            continue
        mag_win = select_mag_window(plates, disturb, errors, joint,
                                    exclude=(acc_win['t0'], acc_win['t1']))
        if mag_win is None:
            print(f"  {name}: no steady-motion magnetic excursion — skipping")
            continue

        print(f"  {name}: selected.")
        print(f"    acc window t={acc_win['t0']:.1f}-{acc_win['t1']:.1f}s, onset t={acc_win['onset_t']:.1f}s "
              f"(acc dev {acc_win['still_deg']:.1f} -> {acc_win['moving_deg']:.1f} deg; "
              f"oracle gap {acc_win['score']:.1f} deg)")
        print(f"    mag window t={mag_win['t0']:.1f}-{mag_win['t1']:.1f}s "
              f"(field dev {mag_win['low_deg']:.1f} -> {mag_win['high_deg']:.1f} deg, "
              f"acc IQR {mag_win['acc_iqr_deg']:.1f} deg; oracle gap {mag_win['score']:.1f} deg)")
        return {'subject': subject, 'plates': plates, 'disturbance': disturb, 'errors': errors,
                'acc_window': acc_win, 'mag_window': mag_win}
    return None

# ==============================================================================
# Drawing
# ==============================================================================

def _method_colors() -> Dict[str, tuple]:
    palette = sns.color_palette(DEFAULT_PALETTE, n_colors=len(PANEL_A_METHODS))
    return dict(zip(PANEL_A_METHODS, palette))

def _draw_error_traces(ax, errors: Dict, window: Dict, methods: List[str],
                       colors: Dict[str, tuple]) -> Dict[str, float]:
    """Error growth since the window start, for this panel's ablation pair (plus
    MAJIC for reference).

    Returns each method's absolute error at t0, which the caller reports
    separately — re-referencing is what makes the window's own contribution
    visible, but it also hides where each method started.
    """
    i0, i1 = window['i0'], window['i1']
    start_error = {}
    for method in methods:
        trace = errors.get(method)
        if trace is None:
            continue
        timestamps, error_rot = trace
        growth = drift_since(error_rot, i0)
        start_error[method] = float(np.degrees(error_rot[i0].magnitude()))
        ax.plot(timestamps[i0:i1], growth[i0:i1],
                color=colors.get(method, 'gray'),
                linewidth=1.6 if method == REFERENCE_METHOD else 2.4,
                linestyle='--' if method == REFERENCE_METHOD else '-',
                label=METHOD_LABELS.get(method, method),
                zorder=2 if method == REFERENCE_METHOD else 3)
    ax.set_ylabel('Error growth\nsince $t_0$ (deg)')
    ax.set_xlim(window['t0'], window['t1'])
    ax.margins(x=0)
    return start_error

def _draw_disturbance(ax, disturb: Dict, joint: str, channel: str, timestamps: np.ndarray,
                      t0: float, t1: float, fs: float, smooth_sec: float) -> None:
    """Bottom row: raw (thin) plus smoothed (heavy) deviation for both segments.

    Both are drawn because at the foot the stride-band modulation and the slow
    positional drift are comparable in size — raw alone reads as a band, smoothed
    alone hides the per-step mechanism.
    """
    parent, child = _joint_sensors(joint)
    mask = (timestamps >= t0) & (timestamps <= t1)
    for role, sensor in (('parent', parent), ('child', child)):
        signal = disturb[sensor][channel]
        color = DISTURBANCE_COLORS[role]
        ax.plot(timestamps[mask], signal[mask], color=color, linewidth=0.6, alpha=0.3)
        ax.plot(timestamps[mask], _smooth(signal, smooth_sec, fs)[mask], color=color,
                linewidth=2.0, label=sensor.replace('_imu', ''))
    label = 'Accelerometer' if channel == 'acc' else 'Magnetometer'
    ax.set_ylabel(f'{label}\ndeviation (deg)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t0, t1)
    ax.margins(x=0)
    ax.legend(loc='upper left', ncol=2, fontsize=10)

def figure_2(stats_df: pd.DataFrame, display: Dict, joint: str, activity: str,
             save: bool = True, show: bool = False) -> None:
    metric_col = f"{METRIC}_{METRIC_UNITS}"
    colors = _method_colors()
    subject = display['subject']
    plates, disturb = display['plates'], display['disturbance']
    parent, _ = _joint_sensors(joint)
    timestamps = plates[parent].imu_trace.timestamps
    fs = 1.0 / np.mean(np.diff(timestamps))

    errors = display['errors']

    fig = plt.figure(figsize=(15, 13.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.1, 1.25, 0.85], hspace=0.42, wspace=0.22)

    # --- Panel A: pooled distribution ---
    ax_a = fig.add_subplot(gs[0, :])
    panel_df = stats_df[stats_df['axis'] == AXIS_TO_PLOT]
    results = test_panels({'all': panel_df}, metric_col, 'method', PANEL_A_METHODS)
    draw_distribution(ax_a, panel_df, metric_col, 'method', PANEL_A_METHODS,
                      results['all'].significant_pairs, 'strip', METHOD_LABELS)
    ax_a.set_ylabel(f'{METRIC.upper()} (degrees)')
    ax_a.set_xlabel('')
    # draw_distribution rotates labels for the crowded sweeps it usually serves;
    # four short labels across a full-width panel read better flat.
    ax_a.set_xticklabels([METHOD_LABELS[m] for m in PANEL_A_METHODS], rotation=0, ha='center')
    ax_a.set_title('A   Joint angle error, all subjects / trials / joints',
                   loc='left', fontsize=16)

    zoom_context = f"Subject{subject}, {activity}, {joint.replace('_', ' ')}"

    # --- Panel B: acc failure (field constant, motion varies) ---
    acc_win = display['acc_window']
    ax_b_err = fig.add_subplot(gs[1, 0])
    ax_b_dist = fig.add_subplot(gs[2, 0], sharex=ax_b_err)
    b_start = _draw_error_traces(ax_b_err, errors, acc_win, PANEL_B_METHODS, colors)
    _draw_disturbance(ax_b_dist, disturb, joint, 'acc', timestamps,
                      acc_win['t0'], acc_win['t1'], fs, SMOOTH_SEC)
    for ax in (ax_b_err, ax_b_dist):
        ax.axvspan(acc_win['t0'], acc_win['onset_t'], color='#D8DEE9', alpha=0.5, zorder=0)
        ax.axvline(acc_win['onset_t'], color='#4C566A', linewidth=1.2, linestyle=':', zorder=1)
    ax_b_err.set_title("B   Stationary $\\rightarrow$ movement\n"
                       "Accelerometer departs from gravity", loc='left', fontsize=14)
    ax_b_err.legend(loc='upper left', fontsize=10, ncol=2)
    ax_b_dist.annotate('stationary', xy=(0.01, 0.82), xycoords='axes fraction', fontsize=10,
                       style='italic', color='#4C566A')

    # --- Panel C: mag failure (motion constant, field varies) ---
    mag_win = display['mag_window']
    ax_c_err = fig.add_subplot(gs[1, 1])
    ax_c_dist = fig.add_subplot(gs[2, 1], sharex=ax_c_err)
    c_start = _draw_error_traces(ax_c_err, errors, mag_win, PANEL_C_METHODS, colors)
    _draw_disturbance(ax_c_dist, disturb, joint, 'mag', timestamps,
                      mag_win['t0'], mag_win['t1'], fs, MAG_SMOOTH_SEC)
    ax_c_err.set_title("C   Steady walking, accelerometer already fixed\n"
                       "Magnetometer departs from the global field", loc='left', fontsize=14)
    ax_c_err.legend(loc='upper left', fontsize=10, ncol=2)

    for ax, start, baseline in ((ax_b_err, b_start, BASELINE_METHOD),
                                (ax_c_err, c_start, ACC_ORACLE_METHOD)):
        ax.annotate(f"{METHOD_LABELS[baseline]} absolute error at $t_0$: "
                    f"{start.get(baseline, float('nan')):.0f}°",
                    xy=(0.99, 0.03), xycoords='axes fraction', ha='right', fontsize=9,
                    style='italic', color='#4C566A')
        plt.setp(ax.get_xticklabels(), visible=False)
    # Trial/joint context rides on the shared x-label rather than a figure-level
    # caption, which has nowhere to sit that doesn't collide with panel A.
    for ax in (ax_b_dist, ax_c_dist):
        ax.set_xlabel(f"Time (s)   ·   {zoom_context}")
    for ax in (ax_b_err, ax_c_err, ax_b_dist, ax_c_dist):
        sns.despine(ax=ax)

    if save:
        path = paths.ensure_parent(PLOTS_DIR / 'figure_2_ekf_failure_modes.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"\nSaved {path}")
        _emit_significance_report(significance_report(results, metric_col),
                                  'figure_2_ekf_failure_modes.png', PLOTS_DIR, save=True)
    if show:
        plt.show()
    plt.close(fig)

# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper figure 2: the EKF's two failure modes.")
    parser.add_argument('--subject', help="Pin the display subject (e.g. 07) instead of "
                                          "auto-selecting the median-oracle-gap subject.")
    parser.add_argument('--joint', default=DISPLAY_JOINT, choices=list(JOINTS))
    parser.add_argument('--activity', default=DISPLAY_ACTIVITY)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    stats_df = load_statistics('ekf_oracle_comparison')
    if stats_df is None:
        print(f"Error: {paths.statistics_path('ekf_oracle_comparison')} not found. "
              "Run experiments/ekf_oracle_comparison.py first.")
        return
    for col in [c for c in stats_df.columns if c.endswith('_rad')]:
        stats_df[col.replace('_rad', '_deg')] = np.degrees(stats_df[col])
    stats_df = stats_df[stats_df['method'].isin(PANEL_A_METHODS)]

    print(f"Selecting display subject for {args.activity} / {args.joint}:")
    display = select_display_subject(stats_df, args.activity, args.joint, forced=args.subject)
    if display is None:
        print("No subject could supply both zoom windows.")
        return

    # Panel A pools every joint; the zoom panels name a specific one, so the
    # pooled panel keeps the L/R-merged joint names for readability.
    stats_df = stats_df.assign(joint_name=stats_df['joint_name'].replace(RENAME_JOINTS))
    figure_2(stats_df, display, args.joint, args.activity, show=args.show)

if __name__ == '__main__':
    main()
