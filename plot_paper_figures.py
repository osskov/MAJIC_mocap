"""
Generates paper figures from the statistics files produced by
`generate_method_data_and_stats.py`:
  - data/all_subject_statistics.parquet          (flat columns, one row per
    trial_type/method/joint_name/subject/axis, with *_rad error metrics)
  - data/all_subject_pearson_correlation.parquet (MultiIndex on
    trial_type/method/joint_name/subject/axis, single 'pearson_r' column)
# Note: the 'marker' method is the mocap ground truth used to compute these
# error/correlation stats against, so it never appears as a row value in either
# file — only the IMU-derived methods (ekf, unprojected, mag_off, mag_on,
# mag_adapt_15, mag_adapt_100, mag_adapt_200) do.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

plt.rcParams.update({
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#EEEEEE',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.weight': 'light',
    'axes.labelweight': 'black',
    'axes.titleweight': 'black',
    'axes.titlesize': 22,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
})

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = Path("data")
STATS_PATH = DATA_DIR / "all_subject_statistics.parquet"
PEARSON_PATH = DATA_DIR / "all_subject_pearson_correlation.parquet"
PLOTS_DIR = Path("plots")

SHOW_PLOTS = True
SAVE_PLOTS = True
PALETTE = "Set2"

SUBJECTS_TO_PLOT = [f"Subject{i:02d}" for i in range(1, 12)]

# Display labels for the method keys written by generate_method_data_and_stats.py.
METHOD_LABELS = {
    'ekf': 'EKF',
    'unprojected': 'Unprojected',
    'mag_off': 'Mag Off',
    'mag_off_flat': 'Mag Off (Flat)',
    'mag_off_dyn': 'Mag Off (Dynamic)',
    'mag_off_unnormalized': 'Mag Off (Unnormalized)',
    'mag_off_normalized': 'Mag Off (Normalized)',
    'mag_off_rescaled': 'Mag Off (Rescaled)',
    'mag_on': 'Mag On',
    'mag_adapt_15': 'MAJIC (15)',
    'mag_adapt_100': 'MAJIC (100)',
    'mag_adapt_200': 'MAJIC (200)',
}
METHODS_TO_PLOT = ['ekf', 'mag_off', 'mag_off_flat', 'mag_off_dyn', 'mag_off_unnormalized', 'mag_off_normalized', 'mag_off_rescaled', 'mag_on', 'mag_adapt_15', 'mag_adapt_100', 'mag_adapt_200', 'mag_adapt', 'mag_adapt_dyn']

# Left/right joints are pooled together under a common name; list them
# individually (e.g. 'R_Hip') here instead if that's not desired.
RENAME_JOINTS = {
    'R_Hip': 'Hip', 'L_Hip': 'Hip',
    'R_Knee': 'Knee', 'L_Knee': 'Knee',
    'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle',
}
JOINT_PLOT_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']

# One of: mean, std, rmse, mae, mad, min, q25, median, q75, max.
METRIC = 'median'
METRIC_UNITS = 'deg'  # 'deg' or 'rad'

# Which rotation-vector axis to summarize for the error-metric plots.
# 'MAG' (rotation magnitude) is the usual choice. Pearson correlation has no
# 'MAG' axis; its plots instead average pearson_r across the X/Y/Z axes.
AXIS_TO_PLOT = 'MAG'

# Facet the distribution plot by joint. Set to None for one pooled plot
# across all joints (the heatmap always breaks down by joint regardless).
FACET_BY = ''

# 'strip' (median/IQR whiskers + points), 'box', or 'bar' (mean + 95% CI).
PLOT_STYLE = 'strip'

# --- Helper Functions ---


def _method_order(present_methods: List[str]) -> List[str]:
    return [m for m in METHODS_TO_PLOT if m in present_methods]


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _remove_outliers(df: pd.DataFrame, metric: str, group_cols: List[str]) -> pd.DataFrame:
    """Drops rows more than 1.5*IQR from the quartiles, computed per group."""
    grouped = df.groupby(group_cols)[metric] if group_cols else df[metric]
    q1 = grouped.transform('quantile', 0.25) if group_cols else grouped.quantile(0.25)
    q3 = grouped.transform('quantile', 0.75) if group_cols else grouped.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[metric] >= lower) & (df[metric] <= upper)]


def _run_statistical_analysis(
    df: pd.DataFrame, metric: str, methods_order: List[str], alpha: float = 0.05
) -> List[Tuple[str, str]]:
    """Friedman test across methods, with Holm-Bonferroni-corrected Wilcoxon
    signed-rank post-hoc tests for pairwise significance if it's significant."""
    block_cols = [c for c in ['trial_type', 'joint_name', 'subject'] if c in df.columns]
    if not block_cols:
        return []

    pivot = df.assign(block_id=df[block_cols].astype(str).agg('_'.join, axis=1)) \
              .pivot_table(index='block_id', columns='method', values=metric)
    pivot = pivot.dropna()
    valid_methods = [m for m in methods_order if m in pivot.columns]

    if pivot.shape[0] < 2 or len(valid_methods) < 2:
        return []

    try:
        _, p_friedman = stats.friedmanchisquare(*[pivot[m] for m in valid_methods])
    except ValueError:
        return []
    if p_friedman >= alpha:
        return []

    pairs, p_values = [], []
    for i in range(len(valid_methods)):
        for j in range(i + 1, len(valid_methods)):
            m1, m2 = valid_methods[i], valid_methods[j]
            pairs.append((m1, m2))
            try:
                _, p = stats.wilcoxon(pivot[m1], pivot[m2], alternative='two-sided', zero_method='zsplit')
            except ValueError:
                p = 1.0
            p_values.append(p)

    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_p * np.arange(len(sorted_p), 0, -1)))
    p_adjusted = np.empty_like(adjusted_sorted)
    p_adjusted[order] = adjusted_sorted

    return [pair for pair, p_adj in zip(pairs, p_adjusted) if p_adj < alpha]


def _draw_significance_brackets(
    ax: plt.Axes, order: List[str], significant_pairs: List[Tuple[str, str]], base: pd.Series
) -> None:
    if not significant_pairs or base.dropna().empty:
        return
    y_step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
    y = base.max() + y_step
    for m1, m2 in sorted(significant_pairs, key=lambda p: abs(order.index(p[0]) - order.index(p[1]))):
        if m1 not in order or m2 not in order:
            continue
        x1, x2 = order.index(m1), order.index(m2)
        ax.plot([x1, x1, x2, x2], [y, y + y_step, y + y_step, y], lw=1.5, c='black')
        ax.text((x1 + x2) / 2, y + y_step, '*', ha='center', va='bottom', fontsize=18, fontweight='bold')
        y += 2 * y_step
    ax.set_ylim(ax.get_ylim()[0], y + y_step)


def _draw_distribution(
    ax: plt.Axes, data: pd.DataFrame, y_col: str, order: List[str],
    significant_pairs: List[Tuple[str, str]], plot_type: str, show_labels: bool = True
) -> None:
    """Draws one panel: the chosen plot style, quartile/CI print-out, value
    labels, and significance brackets."""
    color_map = dict(zip(order, sns.color_palette(PALETTE, n_colors=len(order))))
    grouped = data.groupby('method')[y_col]

    if plot_type == 'bar':
        sns.barplot(data=data, x='method', y=y_col, order=order, hue='method', palette=color_map,
                    legend=False, ax=ax, errorbar=('ci', 95), capsize=0.1, zorder=2)
        n = grouped.count().reindex(order)
        t_crit = (n - 1).clip(lower=1).apply(lambda dof: stats.t.ppf(0.975, dof))
        half_ci = grouped.sem().reindex(order) * t_crit
        center = grouped.mean().reindex(order)
        lower, upper = center - half_ci, center + half_ci
    else:
        if plot_type == 'box':
            sns.boxplot(data=data, x='method', y=y_col, order=order, hue='method', palette=color_map,
                        legend=False, ax=ax, showfliers=False, width=0.7, zorder=2)
        sns.stripplot(data=data, x='method', y=y_col, order=order, hue='method', palette=color_map, legend=False,
                      ax=ax, alpha=0.3 if plot_type == 'box' else 0.5, jitter=0.15, zorder=1)
        quartiles = grouped.quantile([0.25, 0.5, 0.75]).unstack().reindex(order)
        lower, center, upper = quartiles[0.25], quartiles[0.5], quartiles[0.75]

    print(pd.DataFrame({'lower': lower, 'center': center, 'upper': upper})
          .rename(index=_method_label).to_string(float_format='%.3f'))

    if show_labels:
        for i, method in enumerate(order):
            if method not in center.index or pd.isna(center[method]):
                continue
            if plot_type == 'strip':
                color = sns.set_hls_values(color_map.get(method, 'gray'), l=0.4)
                ax.hlines([lower[method], upper[method]], i - 0.35, i + 0.35,
                          color=color, linestyle='--', linewidth=1.5, zorder=10)
                ax.hlines(center[method], i - 0.35, i + 0.45, color=color, linewidth=2, zorder=10)
                ax.text(i + 0.45, center[method], f'{center[method]:.2f}',
                        ha='left', va='center', fontsize=11, fontweight='bold', zorder=11)
            else:
                y_pos = upper[method] if pd.notna(upper[method]) else center[method]
                ax.text(i, y_pos, f'{center[method]:.2f}', ha='center', va='bottom',
                        fontsize=11, fontweight='bold', zorder=11)

    _draw_significance_brackets(ax, order, significant_pairs, upper)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_method_label(m) for m in order], rotation=30, ha='right')
    sns.despine(ax=ax)


def _finalize_and_save_plot(fig: plt.Figure, title: str, filename: str, epilog: Optional[str] = None) -> None:
    fig.suptitle(title, fontsize=18, y=1.02, fontweight='bold')
    if epilog:
        fig.text(0.99, 0.01, epilog, ha='right', va='bottom', fontsize=10,
                  fontstyle='italic', transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0.03 if epilog else 0, 1, 0.97])

    if SAVE_PLOTS:
        PLOTS_DIR.mkdir(exist_ok=True)
        path = PLOTS_DIR / filename
        fig.savefig(path, bbox_inches='tight', dpi=600)
        print(f"Saved plot to {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# --- Main Plotting Functions ---


def plot_metric_distribution(
    df: pd.DataFrame, metric: str, plot_type: str = PLOT_STYLE,
    facet_by: Optional[str] = FACET_BY, higher_is_better: bool = False
) -> None:
    """One figure comparing METHODS_TO_PLOT for `metric`, optionally faceted
    (e.g. one panel per joint)."""
    order = _method_order(df['method'].unique())
    if len(order) < 1:
        print(f"No requested methods present for '{metric}'. Skipping distribution plot.")
        return

    group_cols = ['method'] + ([facet_by] if facet_by else [])
    # data = _remove_outliers(df, metric, group_cols)
    data = df

    if facet_by:
        levels = [j for j in JOINT_PLOT_ORDER if j in data[facet_by].unique()] \
            if facet_by == 'joint_name' else sorted(data[facet_by].unique())
        fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, level in zip(axes, levels):
            level_data = data[data[facet_by] == level]
            sig_pairs = _run_statistical_analysis(level_data, metric, order)
            _draw_distribution(ax, level_data, metric, order, sig_pairs, plot_type)
            ax.set_title(str(level))
            ax.set_xlabel('')
        axes[0].set_ylabel(metric)
        for ax in axes[1:]:
            ax.set_ylabel('')
    else:
        fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2, 6))
        sig_pairs = _run_statistical_analysis(data, metric, order)
        _draw_distribution(ax, data, metric, order, sig_pairs, plot_type)
        ax.set_ylabel(metric)
        ax.set_xlabel('')

    plot_kind = {'strip': 'Median + IQR', 'box': 'Boxplot', 'bar': 'Mean ' + u'±' + ' 95% CI'}[plot_type]
    facet_suffix = f" by {facet_by}" if facet_by else ""
    _finalize_and_save_plot(
        fig, f"{metric}{facet_suffix} ({plot_kind})",
        f"distribution_{metric}{('_by_' + facet_by) if facet_by else ''}_{plot_type}.png"
    )


def plot_metric_heatmap(df: pd.DataFrame, metric: str, higher_is_better: bool = False) -> None:
    """Heatmap of mean `metric` by joint (rows) x method (columns), annotated
    with a significance marker vs. the best method in each row."""
    methods = _method_order(df['method'].unique())
    joints = [j for j in JOINT_PLOT_ORDER if j in df['joint_name'].unique()]
    if not methods or not joints:
        print(f"No data to plot for heatmap of '{metric}'.")
        return

    pivot = df.groupby(['joint_name', 'method'])[metric].mean().unstack().reindex(index=joints, columns=methods)
    annot = pivot.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    for joint in joints:
        row = pivot.loc[joint]
        if row.isnull().all():
            continue
        best_method = row.idxmax() if higher_is_better else row.idxmin()
        sig_pairs = _run_statistical_analysis(df[df['joint_name'] == joint], metric, methods)
        for method in methods:
            if method == best_method or pd.isna(pivot.loc[joint, method]):
                continue
            if any({best_method, method} == {p1, p2} for p1, p2 in sig_pairs):
                annot.loc[joint, method] += "*"

    fig, ax = plt.subplots(figsize=(2.1 * len(methods), 1.8 * len(joints)))
    cmap, vmin, vmax = ('vlag', -1.0, 1.0) if 'pearson' in metric else ('Reds', None, None)
    sns.heatmap(pivot, ax=ax, annot=annot, fmt='', annot_kws={'size': 13, 'weight': 'bold'},
                cmap=cmap, vmin=vmin, vmax=vmax,
                cbar_kws={'label': f"Mean {metric}", 'shrink': 0.8})
    ax.grid(False)

    ax.set_ylabel("Joint")
    ax.set_xlabel("Method")
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=30, ha='right')
    ax.tick_params(axis='y', rotation=0)

    _finalize_and_save_plot(
        fig, f"Mean {metric} by Joint and Method", f"heatmap_{metric}.png",
        epilog="* Significantly different from best method in row (p < 0.05, Wilcoxon)"
    )


# --- Main Execution ---


def _load_stats() -> Optional[pd.DataFrame]:
    if not STATS_PATH.exists():
        print(f"Error: statistics file not found at {STATS_PATH}")
        return None
    df = pd.read_parquet(STATS_PATH)
    for col in [c for c in df.columns if c.endswith('_rad')]:
        df[col.replace('_rad', '_deg')] = np.degrees(df[col])
    df['joint_name'] = df['joint_name'].replace(RENAME_JOINTS)
    return df[df['method'].isin(METHODS_TO_PLOT) & df['subject'].isin(SUBJECTS_TO_PLOT)]


def _load_pearson() -> Optional[pd.DataFrame]:
    """Loads per-axis Pearson correlations and averages X/Y/Z into a single
    'pearson_r' per trial_type/method/joint_name/subject."""
    if not PEARSON_PATH.exists():
        print(f"Error: Pearson correlation file not found at {PEARSON_PATH}")
        return None
    df = pd.read_parquet(PEARSON_PATH).reset_index()
    df['joint_name'] = df['joint_name'].replace(RENAME_JOINTS)
    df = df[df['method'].isin(METHODS_TO_PLOT) & df['subject'].isin(SUBJECTS_TO_PLOT)]
    group_cols = [c for c in ['trial_type', 'method', 'joint_name', 'subject'] if c in df.columns]
    return df.groupby(group_cols, as_index=False)['pearson_r'].mean()


def main():
    metric_col = f"{METRIC}_{METRIC_UNITS}"

    stats_df = _load_stats()
    if stats_df is None:
        return
    print(stats_df.head())

    error_df = stats_df[stats_df['axis'] == AXIS_TO_PLOT]
    plot_metric_distribution(error_df, metric_col)
    plot_metric_heatmap(error_df, metric_col)

    pearson_df = _load_pearson()
    if pearson_df is None:
        return
    print(pearson_df.head())

    plot_metric_distribution(pearson_df, 'pearson_r', higher_is_better=True)
    plot_metric_heatmap(pearson_df, 'pearson_r', higher_is_better=True)

    print("\n--- All plotting complete ---")


if __name__ == "__main__":
    main()