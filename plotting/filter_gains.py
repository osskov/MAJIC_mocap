"""
Where the shipped filter tuning comes from, in two figures.

SURFACE (one per dataset). The sweep's pooled RMSE over the two ratios that are actually
free, acc_std/gyro_std and mag_std/gyro_std -- see experiments/filter_gains.py for why there
are two and not three. The heatmap is the mag_on arm; the line panel beside it is mag_off,
which has no mag_ratio because zeroing the magnetometer makes mag_std provably inert. Both
mark the shipped tuning and the tuning that preceded it, because the distance between those
two marks IS the result.

RESIDUALS (one, all datasets). Why the tuning changed, in two panels: how far each channel's
innovation at the true state sits above the sensor-noise figure the filter was tuned at, and
what that does to the two ratios. Absolute stds are deliberately NOT compared -- a common
rescaling of all three leaves the filter identical (measured: 0.001 deg over six decades), so
only the ratios are comparable and only the ratios moved.
"""
import argparse
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import paths
from plotting import utils  # noqa: F401 (applies shared rcParams styling on import)
from experiments.experiment_utils import (DATASET_STDS, load_statistics,
                                          NOMINAL_ACC_MAGNITUDE, NOMINAL_MAG_MAGNITUDE)
from experiments.filter_gains import NORMALIZING_ARMS, per_sensor_sigma

PLOTS_DIR = paths.plots_dir()

# The tuning every dataset ran at before experiments/filter_gains.py. Hard-coded rather than
# imported because it no longer exists in the source -- and a "before" that silently tracks
# "after" would draw the two marks on top of each other and show nothing.
PREVIOUS_STDS = {'gyro_std': 0.0045, 'acc_std': 0.018, 'mag_std': 0.05}

CHANNEL_UNITS = {'gyro': 'rad/s', 'acc': 'm/s$^2$', 'mag': 'field units'}


def _ratios(stds: Dict[str, float]) -> Dict[str, float]:
    return {'acc': stds['acc_std'] / stds['gyro_std'],
            'mag': stds['mag_std'] / stds['gyro_std']}


def _as_arm_units(stds: Dict[str, float], arm: str) -> Dict[str, float]:
    """`stds` in the units that arm's filter consumes.

    The shipped relative arm takes them as they are. A NORMALIZING arm scales each vector
    measurement to unit length, so a physical std has to be divided by that sensor's nominal
    magnitude -- exactly what experiment_utils' `rescale_stds` does. PREVIOUS_STDS is
    deliberately NOT converted for the EKF: the whole point of that mark is that the old
    pipeline fed physical stds to a normalizing filter without converting them."""
    if arm not in NORMALIZING_ARMS:
        return stds
    return {'gyro_std': stds['gyro_std'],
            'acc_std': stds['acc_std'] / NOMINAL_ACC_MAGNITUDE,
            'mag_std': stds['mag_std'] / NOMINAL_MAG_MAGNITUDE}


def _nearest(values: np.ndarray, target: float) -> float:
    """The swept grid point closest to `target` in LOG space -- the grid is log-spaced, so a
    linear nearest-neighbour would snap an off-grid ratio to the wrong decade."""
    return float(values[np.argmin(np.abs(np.log(values) - np.log(target)))])


def plot_surface(dataset: str, arm: str = 'relative', save: bool = True,
                 show: bool = False) -> None:
    surface_path = paths.experiment_dir("filter_gains") / dataset / f"{arm}_surface.csv"
    if not surface_path.exists():
        print(f"Missing {surface_path}. Run "
              f"`python -m experiments.filter_gains --dataset {dataset} --arm {arm}` first.")
        return
    surface = pd.read_csv(surface_path)
    # The EKF arm normalizes its vector measurements, so its axes are in unit-vector space.
    # Both tunings have to be converted the same way `rescale_stds` does before they can be
    # marked on it -- an unconverted mark would sit 9.81x too high on the acc axis and read
    # as a disagreement between the arms where there is none.
    shipped = _ratios(_as_arm_units(DATASET_STDS[dataset], arm))
    previous = _ratios(PREVIOUS_STDS if arm == 'ekf' else _as_arm_units(PREVIOUS_STDS, arm))

    mag_on = surface[surface.mag_mode == 'on']
    mag_off = surface[surface.mag_mode == 'off']
    n_panels = 1 + int(not mag_on.empty) if not mag_off.empty else 1
    fig, axes = plt.subplots(1, max(n_panels, 1),
                             figsize=(7.5 * max(n_panels, 1), 6.0), squeeze=False)
    axes = axes[0]
    panel = 0

    if not mag_on.empty:
        ax = axes[panel]
        panel += 1
        pivot = mag_on.pivot(index='acc_ratio', columns='mag_ratio', values='rmse_deg')
        sns.heatmap(pivot, ax=ax, cmap='Reds', annot=True, fmt='.0f',
                    annot_kws={'size': 7},
                    cbar_kws={'label': 'Pooled RMSE (deg)', 'shrink': 0.8})
        ax.grid(False)
        ax.set_xlabel('mag_std / gyro_std')
        ax.set_ylabel('acc_std / gyro_std')
        ax.set_title('mag_on')
        ax.set_xticklabels([f"{float(t.get_text()):.3g}" for t in ax.get_xticklabels()],
                           rotation=45, ha='right')
        ax.set_yticklabels([f"{float(t.get_text()):.3g}" for t in ax.get_yticklabels()],
                           rotation=0)

        # Cell centres are at (index + 0.5), so a mark has to be placed on the grid's own
        # coordinates rather than on the ratio value.
        rows = list(pivot.index)
        cols = list(pivot.columns)
        for label, ratios, marker, colour in (
                ('shipped', shipped, '*', 'tab:blue'),
                ('previous', previous, 'X', 'k')):
            y = rows.index(_nearest(np.array(rows), ratios['acc'])) + 0.5
            x = cols.index(_nearest(np.array(cols), ratios['mag'])) + 0.5
            ax.plot(x, y, marker, markersize=18 if marker == '*' else 12,
                    markeredgecolor='w', markeredgewidth=1.4, color=colour,
                    label=f"{label} ({ratios['acc']:.0f}, {ratios['mag']:.1f})")
        # frameon/facecolor forced: the shared rcParams turn legend frames off, which over a
        # heatmap leaves the legend's own marker samples floating on the cells as a second,
        # fake pair of shipped/previous marks.
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95,
                  edgecolor='0.3', fontsize=9)

    if not mag_off.empty:
        ax = axes[panel]
        curve = mag_off.sort_values('acc_ratio')
        ax.plot(curve.acc_ratio, curve.rmse_deg, marker='o', label='mag_off')
        if not mag_on.empty:
            best_by_acc = mag_on.groupby('acc_ratio').rmse_deg.min().sort_index()
            ax.plot(best_by_acc.index, best_by_acc.values, marker='.', linestyle='--',
                    label='mag_on (best mag_ratio)')
        for label, ratios, colour in (('shipped', shipped, 'tab:blue'),
                                      ('previous', previous, 'k')):
            ax.axvline(ratios['acc'], color=colour, linestyle=':', linewidth=2,
                       label=f"{label} acc/gyro = {ratios['acc']:.0f}")
        ax.set_xscale('log')
        ax.set_xlabel('acc_std / gyro_std')
        ax.set_ylabel('Pooled RMSE (deg)')
        ax.set_title('mag_off (mag_std is inert)')
        ax.legend()

    arm_note = (
        "Both axes are in UNIT-VECTOR space, because this arm normalizes its vector "
        "measurements; the marks are the physical tuning converted the same way "
        "rescale_stds converts it. For the EKF the 'previous' mark is left unconverted, "
        "because the old pipeline fed physical stds to a normalizing filter without "
        "converting them — that gap is the result."
        if arm in NORMALIZING_ARMS else
        "The heatmap is the mag_on arm; mag_off is drawn as a line because its mag_std is "
        "provably inert.")
    n_trials = int(surface.n_trials.max())
    utils.finalize_and_save_plot(
        fig, f"Filter gain sweep — {dataset} ({arm})",
        f"filter_gain_surface_{dataset}_{arm}.png", PLOTS_DIR,
        epilog=(f"{n_trials} trials, pooled unweighted over trials and joints, "
                f"first 20 s of each trial excluded"),
        caption=(
            "Pooled joint-angle RMSE over the filter's two free tuning ratios. Scaling all "
            "three stds together leaves the steady-state Kalman gain unchanged, so the "
            "tuning is two-dimensional and this surface is complete — the absolute scale is "
            "a free choice, verified inert over six decades. Star marks the shipped tuning, "
            "cross the tuning that preceded it. " + arm_note + " Note the surface says "
            "nothing about whether the filter is GOOD on this dataset, only which tuning is "
            "least bad: on the biplane half every point is above 36 deg."),
        save=save, show=show)


def plot_residuals(datasets, save: bool = True, show: bool = False) -> None:
    """Innovations and ratios side by side.

    The left panel is the measurement: how far each channel's innovation at the true state
    sits above the sensor-noise figure the filter used to be tuned at. The right panel is
    what that does to the filter, and it is the one that matters -- the absolute stds are
    free (a common rescaling changes nothing, see experiments/filter_gains.py --stage scale),
    so a comparison of absolute values would be comparing conventions. Only the ratios are
    comparable, and only the ratios moved."""
    rows = []
    for dataset in datasets:
        residuals = load_statistics(f"filter_gain_residuals_{dataset}")
        if residuals is None:
            print(f"Missing residuals for {dataset}; skipping.")
            continue
        sigma = {ch: per_sensor_sigma(residuals, ch) for ch in ('gyro', 'acc', 'mag')}
        for channel in ('gyro', 'acc', 'mag'):
            if np.isfinite(sigma[channel]):
                rows.append(dict(dataset=dataset, channel=channel,
                                 innovation=sigma[channel],
                                 previous=PREVIOUS_STDS[f'{channel}_std'],
                                 inflation=sigma[channel] / PREVIOUS_STDS[f'{channel}_std']))
        shipped = DATASET_STDS[dataset]
        rows.append(dict(dataset=dataset, channel='acc/gyro',
                         shipped=shipped['acc_std'] / shipped['gyro_std'],
                         previous_ratio=PREVIOUS_STDS['acc_std'] / PREVIOUS_STDS['gyro_std'],
                         measured_ratio=sigma['acc'] / sigma['gyro']))
        rows.append(dict(dataset=dataset, channel='mag/acc',
                         shipped=shipped['mag_std'] / shipped['acc_std'],
                         previous_ratio=PREVIOUS_STDS['mag_std'] / PREVIOUS_STDS['acc_std'],
                         measured_ratio=sigma['mag'] / sigma['acc']))
    if not rows:
        print("No residual statistics found.")
        return
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))

    inflation = table[table.channel.isin(['gyro', 'acc', 'mag'])]
    pivot = inflation.pivot(index='dataset', columns='channel', values='inflation')
    pivot = pivot.reindex(columns=[c for c in ('gyro', 'acc', 'mag') if c in pivot.columns])
    pivot.plot(kind='bar', ax=axes[0], width=0.78)
    axes[0].axhline(1.0, color='k', linestyle='--', linewidth=1.5)
    axes[0].set_yscale('log')
    axes[0].set_ylabel("measured innovation / previous std")
    axes[0].set_xlabel("")
    axes[0].set_title("How far the old tuning was from the measurement")
    axes[0].tick_params(axis='x', rotation=30)
    # Headroom before the legend, not after: both panels are log-scaled, so a legend placed
    # without it lands on top of the tallest bar rather than above it.
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 8)
    axes[0].legend(title="channel", loc='upper center', ncol=3, framealpha=0.95, fontsize=9)

    ratios = table[table.channel.isin(['acc/gyro', 'mag/acc'])]
    labels = [f"{r.dataset}\n{r.channel}" for r in ratios.itertuples()]
    x = np.arange(len(ratios))
    axes[1].bar(x - 0.2, ratios.previous_ratio, width=0.4, color='k', label='previous')
    axes[1].bar(x + 0.2, ratios.shipped, width=0.4, color='tab:blue', label='shipped')
    axes[1].plot(x + 0.2, ratios.measured_ratio, 'o', color='tab:red', linestyle='none',
                 label='implied by the innovations alone')
    axes[1].set_yscale('log')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel("ratio")
    axes[1].set_title("The two ratios — the only part the filter sees")
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 12)
    axes[1].legend(loc='upper center', ncol=3, framealpha=0.95, fontsize=9)

    utils.finalize_and_save_plot(
        fig, "Why the gains changed", "filter_gain_residuals.png", PLOTS_DIR,
        epilog="innovations are medians over trials; acc/mag are pair differences / sqrt(2)",
        caption=(
            "LEFT: the innovation the filter would see if its state were exactly right — "
            "measured by substituting the mocap rotations into the filter's own measurement "
            "equation — divided by the std the filter used to be tuned with. R does not "
            "model sensor noise, it models everything the measurement equation gets wrong, "
            "so the dashed line is where a correctly tuned channel would sit. The "
            "accelerometer is off by ~140x and the magnetometer by only ~2x. RIGHT: being "
            "wrong by different factors per channel is what put the RATIOS out, and the "
            "ratios are the only thing that reaches the filter — scaling all three stds "
            "together leaves the Kalman gain exactly unchanged. The accelerometer is now "
            "distrusted 5.4x more relative to the gyro and the magnetometer trusted 28x "
            "more relative to the accelerometer. The shipped mag/acc sits above the value "
            "the innovations alone imply because the magnetometer's error is a second-long "
            "correlated bias, not the white noise a Kalman filter assumes. On the biplane "
            "rows the innovations are measurable only inside the 0.5 s fluoroscopic window, "
            "i.e. only during impact, so they are upper bounds."),
        save=save, show=show)


def main():
    parser = argparse.ArgumentParser(description="Plot the filter gain sweep and residuals.")
    parser.add_argument("--datasets", nargs='+',
                        default=['alborno', 'imove', 'imove_biplane', 'imove_biplane_vicon'])
    parser.add_argument("--arms", nargs='+', default=['relative', 'ekf'])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        for arm in args.arms:
            plot_surface(dataset, arm=arm, show=args.show)
    plot_residuals(args.datasets, show=args.show)


if __name__ == '__main__':
    main()
