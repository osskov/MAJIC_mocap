"""
Shared plotting engine for every script in this package. Each script reads
its own results/statistics/<name>_statistics.parquet (see experiments/experiment_utils.py's
save_statistics/load_statistics) and supplies its own x-axis grouping column
(method, threshold, noise combo, ...), display labels, and plot order — this
module only knows how to draw a distribution/heatmap comparison and run the
significance testing behind it, with zero knowledge of what's being compared.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import paths

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

DEFAULT_PALETTE = "Set2"
DEFAULT_JOINT_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']

# ==============================================================================
# Helpers
# ==============================================================================

def order_present(present: List[str], order: List[str]) -> List[str]:
    """Filters `order` down to the values actually present, preserving `order`'s sequence."""
    return [x for x in order if x in present]


def remove_outliers(df: pd.DataFrame, metric: str, group_cols: List[str]) -> pd.DataFrame:
    """Drops rows more than 1.5*IQR from the quartiles, computed per group."""
    grouped = df.groupby(group_cols)[metric] if group_cols else df[metric]
    q1 = grouped.transform('quantile', 0.25) if group_cols else grouped.quantile(0.25)
    q3 = grouped.transform('quantile', 0.75) if group_cols else grouped.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[metric] >= lower) & (df[metric] <= upper)]


def run_statistical_analysis(
    df: pd.DataFrame, metric: str, group_col: str, group_order: List[str],
    block_cols: Optional[List[str]] = None, alpha: float = 0.05
) -> List[Tuple[str, str]]:
    """Friedman test across groups, with Holm-Bonferroni-corrected Wilcoxon
    signed-rank post-hoc tests for pairwise significance if it's significant."""
    if block_cols is None:
        block_cols = [c for c in ['trial_type', 'joint_name', 'subject'] if c in df.columns]
    if not block_cols:
        return []

    pivot = df.assign(block_id=df[block_cols].astype(str).agg('_'.join, axis=1)) \
              .pivot_table(index='block_id', columns=group_col, values=metric)
    pivot = pivot.dropna()
    valid_groups = [g for g in group_order if g in pivot.columns]

    if pivot.shape[0] < 2 or len(valid_groups) < 2:
        return []

    try:
        _, p_friedman = stats.friedmanchisquare(*[pivot[g] for g in valid_groups])
    except ValueError:
        return []
    if p_friedman >= alpha:
        return []

    pairs, p_values = [], []
    for i in range(len(valid_groups)):
        for j in range(i + 1, len(valid_groups)):
            g1, g2 = valid_groups[i], valid_groups[j]
            pairs.append((g1, g2))
            try:
                _, p = stats.wilcoxon(pivot[g1], pivot[g2], alternative='two-sided', zero_method='zsplit')
            except ValueError:
                p = 1.0
            p_values.append(p)

    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_p * np.arange(len(sorted_p), 0, -1)))
    p_adjusted = np.empty_like(adjusted_sorted)
    p_adjusted[order] = adjusted_sorted

    return [pair for pair, p_adj in zip(pairs, p_adjusted) if p_adj < alpha]


def draw_significance_brackets(
    ax: plt.Axes, order: List[str], significant_pairs: List[Tuple[str, str]], base: pd.Series
) -> None:
    if not significant_pairs or base.dropna().empty:
        return
    y_step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
    y = base.max() + y_step
    for g1, g2 in sorted(significant_pairs, key=lambda p: abs(order.index(p[0]) - order.index(p[1]))):
        if g1 not in order or g2 not in order:
            continue
        x1, x2 = order.index(g1), order.index(g2)
        ax.plot([x1, x1, x2, x2], [y, y + y_step, y + y_step, y], lw=1.5, c='black')
        ax.text((x1 + x2) / 2, y + y_step, '*', ha='center', va='bottom', fontsize=18, fontweight='bold')
        y += 2 * y_step
    ax.set_ylim(ax.get_ylim()[0], y + y_step)


def draw_distribution(
    ax: plt.Axes, data: pd.DataFrame, y_col: str, group_col: str, order: List[str],
    significant_pairs: List[Tuple[str, str]], plot_type: str, labels: Optional[Dict[str, str]] = None,
    palette: str = DEFAULT_PALETTE, show_labels: bool = True
) -> None:
    """Draws one panel: the chosen plot style, quartile/CI print-out, value
    labels, and significance brackets."""
    labels = labels or {}
    label = lambda g: labels.get(g, g)
    color_map = dict(zip(order, sns.color_palette(palette, n_colors=len(order))))
    grouped = data.groupby(group_col)[y_col]

    if plot_type == 'bar':
        sns.barplot(data=data, x=group_col, y=y_col, order=order, hue=group_col, palette=color_map,
                    legend=False, ax=ax, errorbar=('ci', 95), capsize=0.1, zorder=2)
        n = grouped.count().reindex(order)
        t_crit = (n - 1).clip(lower=1).apply(lambda dof: stats.t.ppf(0.975, dof))
        half_ci = grouped.sem().reindex(order) * t_crit
        center = grouped.mean().reindex(order)
        lower, upper = center - half_ci, center + half_ci
    else:
        if plot_type == 'box':
            sns.boxplot(data=data, x=group_col, y=y_col, order=order, hue=group_col, palette=color_map,
                        legend=False, ax=ax, showfliers=False, width=0.7, zorder=2)
        sns.stripplot(data=data, x=group_col, y=y_col, order=order, hue=group_col, palette=color_map, legend=False,
                      ax=ax, alpha=0.3 if plot_type == 'box' else 0.5, jitter=0.15, zorder=1)
        quartiles = grouped.quantile([0.25, 0.5, 0.75]).unstack().reindex(order)
        lower, center, upper = quartiles[0.25], quartiles[0.5], quartiles[0.75]

    print(pd.DataFrame({'lower': lower, 'center': center, 'upper': upper})
          .rename(index=label).to_string(float_format='%.3f'))

    if show_labels:
        for i, group in enumerate(order):
            if group not in center.index or pd.isna(center[group]):
                continue
            if plot_type == 'strip':
                color = sns.set_hls_values(color_map.get(group, 'gray'), l=0.4)
                ax.hlines([lower[group], upper[group]], i - 0.35, i + 0.35,
                          color=color, linestyle='--', linewidth=1.5, zorder=10)
                ax.hlines(center[group], i - 0.35, i + 0.45, color=color, linewidth=2, zorder=10)
                ax.text(i + 0.45, center[group], f'{center[group]:.2f}',
                        ha='left', va='center', fontsize=11, fontweight='bold', zorder=11)
            else:
                y_pos = upper[group] if pd.notna(upper[group]) else center[group]
                ax.text(i, y_pos, f'{center[group]:.2f}', ha='center', va='bottom',
                        fontsize=11, fontweight='bold', zorder=11)

    draw_significance_brackets(ax, order, significant_pairs, upper)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label(g) for g in order], rotation=30, ha='right')
    sns.despine(ax=ax)


def finalize_and_save_plot(
    fig: plt.Figure, title: str, filename: str, plots_dir: Path = paths.PLOTS_DIR,
    epilog: Optional[str] = None, save: bool = True, show: bool = True
) -> None:
    fig.suptitle(title, fontsize=18, y=1.02, fontweight='bold')
    if epilog:
        fig.text(0.99, 0.01, epilog, ha='right', va='bottom', fontsize=10,
                  fontstyle='italic', transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0.03 if epilog else 0, 1, 0.97])

    if save:
        path = paths.ensure_parent(plots_dir / filename)
        fig.savefig(path, bbox_inches='tight', dpi=600)
        print(f"Saved plot to {path}")
    if show:
        plt.show()
    plt.close(fig)

# ==============================================================================
# Main plotting functions
# ==============================================================================

def plot_metric_distribution(
    df: pd.DataFrame, metric: str, group_col: str, group_order: List[str], plots_dir: Path,
    labels: Optional[Dict[str, str]] = None, plot_type: str = 'strip', facet_by: Optional[str] = None,
    facet_order: Optional[List[str]] = None, palette: str = DEFAULT_PALETTE, save: bool = True, show: bool = True
) -> None:
    """One figure comparing `group_order` for `metric`, optionally faceted (e.g.
    one panel per joint)."""
    order = order_present(df[group_col].unique(), group_order)
    if len(order) < 1:
        print(f"No requested {group_col} values present for '{metric}'. Skipping distribution plot.")
        return

    if facet_by:
        levels = order_present(df[facet_by].unique(), facet_order) if facet_order \
            else sorted(df[facet_by].unique())
        fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, level in zip(axes, levels):
            level_data = df[df[facet_by] == level]
            sig_pairs = run_statistical_analysis(level_data, metric, group_col, order)
            draw_distribution(ax, level_data, metric, group_col, order, sig_pairs, plot_type, labels, palette)
            ax.set_title(str(level))
            ax.set_xlabel('')
        axes[0].set_ylabel(metric)
        for ax in axes[1:]:
            ax.set_ylabel('')
    else:
        fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2, 6))
        sig_pairs = run_statistical_analysis(df, metric, group_col, order)
        draw_distribution(ax, df, metric, group_col, order, sig_pairs, plot_type, labels, palette)
        ax.set_ylabel(metric)
        ax.set_xlabel('')

    plot_kind = {'strip': 'Median + IQR', 'box': 'Boxplot', 'bar': 'Mean ' + u'±' + ' 95% CI'}[plot_type]
    facet_suffix = f" by {facet_by}" if facet_by else ""
    finalize_and_save_plot(
        fig, f"{metric}{facet_suffix} ({plot_kind})",
        f"distribution_{metric}{('_by_' + facet_by) if facet_by else ''}_{plot_type}.png",
        plots_dir, save=save, show=show
    )


def plot_metric_heatmap(
    df: pd.DataFrame, metric: str, group_col: str, group_order: List[str], plots_dir: Path,
    labels: Optional[Dict[str, str]] = None, joint_order: List[str] = DEFAULT_JOINT_ORDER,
    higher_is_better: bool = False, save: bool = True, show: bool = True
) -> None:
    """Heatmap of mean `metric` by joint (rows) x `group_col` (columns), annotated
    with a significance marker vs. the best value in each row."""
    labels = labels or {}
    groups = order_present(df[group_col].unique(), group_order)
    joints = [j for j in joint_order if j in df['joint_name'].unique()]
    if not groups or not joints:
        print(f"No data to plot for heatmap of '{metric}'.")
        return

    pivot = df.groupby(['joint_name', group_col])[metric].mean().unstack().reindex(index=joints, columns=groups)
    annot = pivot.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    for joint in joints:
        row = pivot.loc[joint]
        if row.isnull().all():
            continue
        best_group = row.idxmax() if higher_is_better else row.idxmin()
        sig_pairs = run_statistical_analysis(df[df['joint_name'] == joint], metric, group_col, groups)
        for group in groups:
            if group == best_group or pd.isna(pivot.loc[joint, group]):
                continue
            if any({best_group, group} == {g1, g2} for g1, g2 in sig_pairs):
                annot.loc[joint, group] += "*"

    fig, ax = plt.subplots(figsize=(2.1 * len(groups), 1.8 * len(joints)))
    sns.heatmap(pivot, ax=ax, annot=annot, fmt='', annot_kws={'size': 13, 'weight': 'bold'},
                cmap='Reds', cbar_kws={'label': f"Mean {metric}", 'shrink': 0.8})
    ax.grid(False)

    ax.set_ylabel("Joint")
    ax.set_xlabel(group_col)
    ax.set_xticklabels([labels.get(g, g) for g in groups], rotation=30, ha='right')
    ax.tick_params(axis='y', rotation=0)

    finalize_and_save_plot(
        fig, f"Mean {metric} by Joint and {group_col}", f"heatmap_{metric}.png", plots_dir,
        epilog="* Significantly different from best value in row (p < 0.05, Wilcoxon)", save=save, show=show
    )
