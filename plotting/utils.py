"""
Shared plotting engine for every script in this package. Each script reads
its own results/statistics/<name>_statistics.parquet (see experiments/experiment_utils.py's
save_statistics/load_statistics) and supplies its own x-axis grouping column
(method, threshold, noise combo, ...), display labels, and plot order — this
module only knows how to draw a distribution/heatmap comparison and run the
significance testing behind it, with zero knowledge of what's being compared.
"""
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    'axes.grid': False,
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


# ==============================================================================
# Significance testing
# ==============================================================================
# Two decisions here are easy to get wrong, so both are spelled out.
#
# 1. WHAT COUNTS AS A REPLICATE. What this repo measures is a filter's ability to
#    estimate a relative orientation, so the replicate is a sensor pair spanning a
#    joint, not a person. Which dimensions actually replicate was settled by
#    measuring the intraclass correlation of the paired method difference on this
#    dataset, rather than argued from first principles:
#
#      different joints, same subject          ICC 0.011 - 0.068   -> independent
#      same sensor pair, walking vs complex    ICC 0.45  - 0.58    -> NOT independent
#      left vs right of a joint, same subject  ICC 0.51  - 0.56    -> NOT independent
#
#    So joints do count separately. The two activities are the same sensors in the
#    same places on the same person, and the two sides of a joint track each other
#    about as closely, so both are averaged into the cell instead of being counted
#    twice — counting either would inflate n by a design effect of roughly 1.5.
#
#    DEFAULT_BLOCK_COLS is therefore ['subject', 'joint_name']: 11 x 4 = 44 blocks
#    pooled, and 11 inside any single-joint panel. This assumes joint_name has been
#    collapsed to joint TYPE (Hip, not R_Hip/L_Hip); _warn_if_sides_split flags
#    callers that forget, since leaving sides split silently overstates precision.
#
# 2. A FIGURE IS ONE CORRECTION FAMILY, NOT ONE PANEL. Correcting within each panel
#    and then drawing five panels side by side is still five uncorrected families.
#    So: test every panel -> Holm across the panels' omnibus p-values -> Holm across
#    the pooled pairwise p-values of the panels that survived.
#
# At n=44 a consistent but physically trivial difference will clear p<0.05, so every
# result carries its mean difference in the metric's own units and a rank-biserial
# effect size. Significance without magnitude is not reportable.

DEFAULT_BLOCK_COLS = ['subject', 'joint_name']

_SIDE_PREFIXED = re.compile(r'^[RL]_')


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, returned in the input's order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    weights = np.arange(len(p_values), 0, -1)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(p_values[order] * weights))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the effect size that pairs with
    Wilcoxon. Ranges -1..+1; positive means x tends to exceed y. Exact ties are
    dropped, per convention and irrelevant for continuous error metrics."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def _warn_if_sides_split(df: pd.DataFrame, block_cols: List[str]) -> None:
    """Left and right of a joint correlate at ICC ~0.5, so they are one replicate,
    not two. Blocking on a joint_name that still carries R_/L_ prefixes silently
    doubles the block count and overstates precision by ~1.5x."""
    if 'joint_name' not in block_cols or 'joint_name' not in df.columns:
        return
    split = sorted({j for j in df['joint_name'].dropna().unique() if _SIDE_PREFIXED.match(str(j))})
    if split:
        print(f"Warning: joint_name still has side-split values {split}. Collapse L/R to joint "
              f"type (see RENAME_JOINTS in the figure scripts) — blocking on them treats the two "
              f"sides of a joint as independent replicates, which the data does not support.")


@dataclass
class PanelTest:
    """One panel's test result. Correction fields are filled in by `test_panels`."""
    key: str
    groups: List[str]
    n_blocks: int
    friedman_p: float = float('nan')
    kendalls_w: float = float('nan')
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    pair_p_raw: List[float] = field(default_factory=list)
    pair_mean_diff: List[float] = field(default_factory=list)
    pair_effect: List[float] = field(default_factory=list)
    friedman_p_adj: float = float('nan')
    pair_p_adj: List[float] = field(default_factory=list)
    significant_pairs: List[Tuple[str, str]] = field(default_factory=list)
    skipped_reason: Optional[str] = None


def block_pivot(df: pd.DataFrame, metric: str, group_col: str,
                block_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Reduces a panel to one row per block and one column per group.

    Any dimension not named in block_cols (side, activity, ...) is averaged into the
    block's value. That averaging is stated explicitly here rather than left to
    pivot_table's default aggfunc, because it is a real statistical decision, not a
    formatting one. Blocks missing any group are dropped — Friedman needs complete
    blocks."""
    block_cols = DEFAULT_BLOCK_COLS if block_cols is None else block_cols
    block_cols = [c for c in block_cols if c in df.columns]
    if not block_cols:
        return pd.DataFrame()
    return (df.assign(_block=df[block_cols].astype(str).agg('_'.join, axis=1))
              .pivot_table(index='_block', columns=group_col, values=metric, aggfunc='mean')
              .dropna())


def test_panel(df: pd.DataFrame, metric: str, group_col: str, group_order: List[str],
               key: str = "", block_cols: Optional[List[str]] = None) -> PanelTest:
    """Friedman omnibus + pairwise Wilcoxon for one panel. Returns RAW p-values;
    correction happens in `test_panels`, across every panel in the figure."""
    pivot = block_pivot(df, metric, group_col, block_cols)
    groups = [g for g in group_order if g in pivot.columns]
    n = int(pivot.shape[0])

    if n < 2 or len(groups) < 2:
        return PanelTest(key=key, groups=groups, n_blocks=n,
                         skipped_reason=(f"need >=2 blocks and >=2 groups "
                                         f"(got {n} blocks, {len(groups)} groups)"))
    try:
        chi2, p_friedman = stats.friedmanchisquare(*[pivot[g] for g in groups])
    except ValueError as e:
        return PanelTest(key=key, groups=groups, n_blocks=n, skipped_reason=f"Friedman failed: {e}")

    # Kendall's W: the Friedman statistic rescaled to 0..1 agreement across blocks.
    kendalls_w = float(chi2 / (n * (len(groups) - 1)))

    pairs, p_raw, mean_diffs, effects = [], [], [], []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            pairs.append((g1, g2))
            try:
                _, p = stats.wilcoxon(pivot[g1], pivot[g2], alternative='two-sided', zero_method='zsplit')
            except ValueError:
                p = 1.0
            p_raw.append(float(p))
            mean_diffs.append(float((pivot[g1] - pivot[g2]).mean()))
            effects.append(_rank_biserial(pivot[g1].to_numpy(), pivot[g2].to_numpy()))

    return PanelTest(key=key, groups=groups, n_blocks=n, friedman_p=float(p_friedman),
                     kendalls_w=kendalls_w, pairs=pairs, pair_p_raw=p_raw,
                     pair_mean_diff=mean_diffs, pair_effect=effects)


def test_panels(panels: Dict[str, pd.DataFrame], metric: str, group_col: str, group_order: List[str],
                block_cols: Optional[List[str]] = None, alpha: float = 0.05) -> Dict[str, PanelTest]:
    """Tests every panel of a figure as a single Holm family.

    Holm is applied twice: once across the panels' Friedman p-values (the omnibus
    family), then once across the pooled pairwise p-values of the panels that
    survived it. Pass a one-entry dict for an unfaceted figure — the arithmetic then
    degenerates to the ordinary single-family case."""
    for df in panels.values():
        _warn_if_sides_split(df, DEFAULT_BLOCK_COLS if block_cols is None else block_cols)
        break

    results = {key: test_panel(df, metric, group_col, group_order, key=key, block_cols=block_cols)
               for key, df in panels.items()}

    testable = [r for r in results.values() if r.skipped_reason is None]
    if not testable:
        return results

    for result, p_adj in zip(testable, holm_bonferroni([r.friedman_p for r in testable])):
        result.friedman_p_adj = float(p_adj)

    survivors = [r for r in testable if r.friedman_p_adj < alpha]
    flat_p = [p for r in survivors for p in r.pair_p_raw]
    if not flat_p:
        return results

    flat_adj = holm_bonferroni(flat_p)
    cursor = 0
    for result in survivors:
        n_pairs = len(result.pair_p_raw)
        result.pair_p_adj = [float(p) for p in flat_adj[cursor:cursor + n_pairs]]
        cursor += n_pairs
        result.significant_pairs = [pair for pair, p_adj in zip(result.pairs, result.pair_p_adj)
                                    if p_adj < alpha]
    return results


def significance_report(results: Dict[str, PanelTest], metric: str) -> pd.DataFrame:
    """Flattens test results into a citable table: n, effect magnitude, both
    corrected p-values, and effect size."""
    rows = []
    for key, r in results.items():
        if r.skipped_reason is not None:
            rows.append({'panel': key, 'n_blocks': r.n_blocks, 'comparison': '—',
                         'note': r.skipped_reason})
            continue
        for idx, (g1, g2) in enumerate(r.pairs):
            rows.append({
                'panel': key, 'metric': metric, 'n_blocks': r.n_blocks,
                'comparison': f"{g1} vs {g2}",
                'mean_diff': r.pair_mean_diff[idx],
                'friedman_p': r.friedman_p, 'friedman_p_holm': r.friedman_p_adj,
                'kendalls_w': r.kendalls_w,
                'wilcoxon_p': r.pair_p_raw[idx],
                'wilcoxon_p_holm': r.pair_p_adj[idx] if r.pair_p_adj else float('nan'),
                'rank_biserial': r.pair_effect[idx],
                'significant': (g1, g2) in r.significant_pairs,
            })
    return pd.DataFrame(rows)


def run_statistical_analysis(
    df: pd.DataFrame, metric: str, group_col: str, group_order: List[str],
    block_cols: Optional[List[str]] = None, alpha: float = 0.05
) -> List[Tuple[str, str]]:
    """Single-panel convenience wrapper for callers that only want the pairs.
    Equivalent to a `test_panels` family of one."""
    results = test_panels({'': df}, metric, group_col, group_order, block_cols=block_cols, alpha=alpha)
    return results[''].significant_pairs


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

    if plot_type in ('bar', 'bar_sd'):
        # 'bar' shows the precision of the mean (95% CI), 'bar_sd' the spread of the
        # data itself (±1 SD) — the latter is what a "mean and std" table reports.
        sns.barplot(data=data, x=group_col, y=y_col, order=order, hue=group_col, palette=color_map,
                    legend=False, ax=ax, errorbar='sd' if plot_type == 'bar_sd' else ('ci', 95),
                    capsize=0.1, zorder=2)
        center = grouped.mean().reindex(order)
        if plot_type == 'bar_sd':
            half = grouped.std().reindex(order)
        else:
            n = grouped.count().reindex(order)
            t_crit = (n - 1).clip(lower=1).apply(lambda dof: stats.t.ppf(0.975, dof))
            half = grouped.sem().reindex(order) * t_crit
        lower, upper = center - half, center + half
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


def _significance_epilog(results: Dict[str, PanelTest], alpha: float) -> str:
    """One line describing the test behind the brackets, for the figure footer."""
    testable = [r for r in results.values() if r.skipped_reason is None]
    if not testable:
        return "No significance testing (insufficient blocks or groups)"
    n_blocks = {r.n_blocks for r in testable}
    n_text = f"n={n_blocks.pop()}" if len(n_blocks) == 1 else "n varies by panel"
    scope = " across all panels" if len(testable) > 1 else ""
    return (f"* Friedman + Wilcoxon signed-rank, Holm-corrected{scope} (p < {alpha:g}); "
            f"blocked by subject x joint, averaged over side and activity, {n_text}")


def _emit_significance_report(report: pd.DataFrame, figure_name: str, plots_dir: Path,
                              save: bool = True) -> None:
    """Prints the test table and, when saving, writes it beside the figure.

    The previous implementation computed adjusted p-values and discarded them,
    keeping one boolean per pair — so a figure asserted significance that could not
    be quoted or checked. These are the numbers that belong in the paper text."""
    if report.empty:
        return
    columns = [c for c in ['panel', 'n_blocks', 'comparison', 'mean_diff', 'friedman_p_holm',
                           'wilcoxon_p_holm', 'rank_biserial', 'significant', 'note']
               if c in report.columns]
    print(report[columns].to_string(index=False, float_format='%.4f'))
    if save:
        path = paths.ensure_parent(plots_dir / f"{Path(figure_name).stem}_stats.csv")
        report.to_csv(path, index=False)
        print(f"Saved significance report to {path}")


# Characters per line in a rendered caption, at CAPTION_FONTSIZE. Wrapping is done here
# rather than left to the caller because a caption written as one long string must not
# decide the figure's width -- `bbox_inches='tight'` would expand the canvas to fit the
# text and leave the axes crushed into a corner, which is exactly what a one-line epilog
# did to the two-column Al Borno grid.
CAPTION_WIDTH_PER_INCH = 15.5
CAPTION_FONTSIZE = 9.5
# Most of the figure a caption may claim. See the clamp in finalize_and_save_plot.
CAPTION_MAX_BAND = 0.4


class MissingCaptionWarning(UserWarning):
    """Raised as a warning when a figure is saved with no caption. Its own class so a caller
    that genuinely wants an uncaptioned figure can silence exactly this and nothing else."""


def finalize_and_save_plot(
    fig: plt.Figure, title: str, filename: str, plots_dir: Path = paths.PLOTS_DIR,
    epilog: Optional[str] = None, save: bool = True, show: bool = True,
    caption: Optional[str] = None
) -> None:
    """Title, optional caption and epilog, then save.

    EVERY FIGURE SHOULD PASS A CAPTION. A figure travels: it goes into a slide, a message, a
    paper draft, and it arrives without the code or the report section that explains it. The
    caption is what makes it readable on its own -- what is plotted, what the axes mean, what
    the reader is supposed to conclude, and what the figure does NOT show. `epilog` is a
    different thing and both can be used: it is the one-line provenance stamp (how many
    trials, which legend marks mean what), not an explanation.

    Absent, a warning names the file. Not an exception, because a figure that renders without
    a caption is still worth having and failing the run would throw away the plot as well as
    the caption -- but the warning is deliberately loud enough to be fixed.
    """
    fig.suptitle(title, fontsize=18, y=1.02, fontweight='bold')

    # Caption at the bottom, epilog stacked on the line above it. Both are measured in figure
    # fractions off the actual font size and figure height rather than guessed: a fixed y put
    # the epilog straight through the middle of a five-line caption on the 13x5 panels.
    line_height = (CAPTION_FONTSIZE * 1.35) / (fig.get_figheight() * 72.0)
    caption_height = 0.0

    if caption:
        width = int(fig.get_figwidth() * CAPTION_WIDTH_PER_INCH)
        wrapped = textwrap.fill(' '.join(caption.split()), width=max(width, 60))
        fig.text(0.01, 0.005, wrapped, ha='left', va='bottom', fontsize=CAPTION_FONTSIZE,
                 transform=fig.transFigure)
        caption_height = (wrapped.count('\n') + 1) * line_height
    else:
        warnings.warn(
            f"{filename} was saved without a caption. A figure travels without the code that "
            f"made it; pass caption= describing what is plotted and what to conclude.",
            MissingCaptionWarning, stacklevel=2)

    reserved = caption_height + 0.01 if caption else 0.0
    if epilog:
        fig.text(0.99, reserved + 0.005, epilog, ha='right', va='bottom',
                 fontsize=10, fontstyle='italic', transform=fig.transFigure)
        reserved += 1.6 * line_height
    # Clamped, because a rect that leaves no room for the axes makes tight_layout give up
    # entirely and warn -- which would lose the layout for the whole figure, not just the
    # caption. Past this point the text runs into the axes, which is the better failure: it
    # is visible, and it means the caption wants shortening.
    fig.tight_layout(rect=[0, min(max(reserved, 0.03 if epilog else 0.0), CAPTION_MAX_BAND),
                           1, 0.97])

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
    facet_order: Optional[List[str]] = None, palette: str = DEFAULT_PALETTE, save: bool = True, show: bool = True,
    block_cols: Optional[List[str]] = None, alpha: float = 0.05,
    ylabel: Optional[str] = None, title: Optional[str] = None, filename: Optional[str] = None,
    caption: Optional[str] = None
) -> None:
    """One figure comparing `group_order` for `metric`, optionally faceted (e.g.
    one panel per joint).

    All panels are tested as one Holm family (see test_panels), blocking on
    `block_cols` (default: subject x joint). A `<figure>_stats.csv` with n, effect
    magnitudes and corrected p-values is written beside the figure.

    `ylabel`/`title`/`filename` override the names derived from `metric` — paper
    figures want a typeset axis label and a stable `figure_N_*.png`, exploratory
    sweeps are happy with the defaults."""
    order = order_present(df[group_col].unique(), group_order)
    if len(order) < 1:
        print(f"No requested {group_col} values present for '{metric}'. Skipping distribution plot.")
        return

    if facet_by:
        levels = order_present(df[facet_by].unique(), facet_order) if facet_order \
            else sorted(df[facet_by].unique())
        panels = {str(level): df[df[facet_by] == level] for level in levels}
    else:
        panels = {'all': df}

    results = test_panels(panels, metric, group_col, order, block_cols=block_cols, alpha=alpha)

    if facet_by:
        fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, level in zip(axes, levels):
            draw_distribution(ax, panels[str(level)], metric, group_col, order,
                              results[str(level)].significant_pairs, plot_type, labels, palette)
            ax.set_title(str(level))
            ax.set_xlabel('')
        axes[0].set_ylabel(ylabel or metric)
        for ax in axes[1:]:
            ax.set_ylabel('')
    else:
        fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2, 6))
        draw_distribution(ax, df, metric, group_col, order, results['all'].significant_pairs,
                          plot_type, labels, palette)
        ax.set_ylabel(ylabel or metric)
        ax.set_xlabel('')

    plot_kind = {'strip': 'Median + IQR', 'box': 'Boxplot',
                 'bar': 'Mean ' + u'±' + ' 95% CI', 'bar_sd': 'Mean ' + u'±' + ' SD'}[plot_type]
    facet_suffix = f" by {facet_by}" if facet_by else ""
    out_name = filename or f"distribution_{metric}{('_by_' + facet_by) if facet_by else ''}_{plot_type}.png"
    finalize_and_save_plot(
        fig, title if title is not None else f"{metric}{facet_suffix} ({plot_kind})", out_name, plots_dir,
        epilog=_significance_epilog(results, alpha), save=save, show=show, caption=caption
    )
    _emit_significance_report(significance_report(results, metric), out_name, plots_dir, save=save)


def plot_metric_heatmap(
    df: pd.DataFrame, metric: str, group_col: str, group_order: List[str], plots_dir: Path,
    labels: Optional[Dict[str, str]] = None, joint_order: List[str] = DEFAULT_JOINT_ORDER,
    higher_is_better: bool = False, save: bool = True, show: bool = True,
    block_cols: Optional[List[str]] = None, alpha: float = 0.05,
    caption: Optional[str] = None, filename: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """Heatmap of mean `metric` by joint (rows) x `group_col` (columns), annotated
    with a significance marker vs. the best value in each row.

    Every row is one panel of a single Holm family, so a star means "differs from
    this row's best after correcting across the whole heatmap", not just across the
    row. Writes a `<figure>_stats.csv` beside the figure.

    `filename`/`title` override the names derived from `metric`, which a caller drawing ONE
    HEATMAP PER ERROR AXIS needs: the derived name is `heatmap_<metric>.png` for all of them, so
    without an override the three anatomical axes overwrite each other and the last one wins."""
    labels = labels or {}
    groups = order_present(df[group_col].unique(), group_order)
    joints = [j for j in joint_order if j in df['joint_name'].unique()]
    if not groups or not joints:
        print(f"No data to plot for heatmap of '{metric}'.")
        return
    dropped = sorted(set(df['joint_name'].unique()) - set(joints))
    if dropped:
        print(f"Warning: heatmap is indexed by joint type and dropped {dropped}. "
              f"Collapse L/R to joint type (see RENAME_JOINTS) to include them.")

    pivot = df.groupby(['joint_name', group_col])[metric].mean().unstack().reindex(index=joints, columns=groups)
    annot = pivot.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    results = test_panels({j: df[df['joint_name'] == j] for j in joints},
                          metric, group_col, groups, block_cols=block_cols, alpha=alpha)

    for joint in joints:
        row = pivot.loc[joint]
        if row.isnull().all():
            continue
        best_group = row.idxmax() if higher_is_better else row.idxmin()
        sig_pairs = results[joint].significant_pairs
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

    out_name = filename or f"heatmap_{metric}.png"
    n_blocks = {r.n_blocks for r in results.values() if r.skipped_reason is None}
    n_text = f"n={n_blocks.pop()}" if len(n_blocks) == 1 else "n varies by row"
    finalize_and_save_plot(
        fig, title if title is not None else f"Mean {metric} by Joint and {group_col}",
        out_name, plots_dir,
        epilog=(f"* Differs from best in row (Wilcoxon signed-rank, Holm-corrected across the "
                f"whole heatmap, p < {alpha:g}, {n_text} subjects)"),
        save=save, show=show, caption=caption
    )
    _emit_significance_report(significance_report(results, metric), out_name, plots_dir, save=save)
