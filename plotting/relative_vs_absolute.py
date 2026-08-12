"""
Figure for experiments/relative_vs_absolute.py: why a relative correction at the joint center
beats correcting each sensor against a global reference, as geometry and then as a measurement.

Everything is read back from that experiment's per-trial tables — nothing here reloads raw
data, re-projects or re-computes an angle, so no number in the figure can disagree with
results/statistics/relative_vs_absolute_statistics.parquet.

Two arguments, and the figure keeps them apart on purpose
--------------------------------------------------------
The experiment measures two different things, whose effect sizes differ by two orders of
magnitude. Drawing them as one result would be the main way to misread this figure, so the
rows are split by argument rather than by field:

  DIRECTION (rows 1-2, facts 1-4).  How much rotation each residual demands of a single vector
  field. theta_rel is what the relative residual demands, theta_comp what the absolute route's
  two corrections leave in the joint angle, theta_sum what they cost across both segments.
  theta_rel <= theta_comp and theta_rel <= theta_sum are exact. The first gap is ~14 deg, the
  second ~0.06 deg, and panel C is what explains the difference: the penalty adds in
  QUADRATURE, so a sub-degree out-of-plane twist against a several-degree base costs nothing.

  INVARIANT (row 3, facts 5-7).  The angle between the accelerometer and magnetometer vectors
  is invariant to rotation, so a mismatch in it is irreducible — no orientation estimate for
  either segment can remove it. Half the mismatch is the achievable floor. This is where the
  joint-angle advantage lives, and it is a factor of two.

Panels
------
A  The spherical triangle, drawn. Defines every angle the other panels use, and the shaded
   area is psi, whose role fact 3 makes exact. ANGLES ARE EXAGGERATED: the real triangles are
   slivers, which is the whole content of panel C, so a to-scale schematic would be a line.

B  theta_comp and theta_sum against theta_rel, with the identity line. The identity line is the
   argument: theta_comp lies ON it (fact 2 is tight in practice — the two absolute errors are
   common mode and cancel in the joint angle) while theta_sum sits far above it (fact 1 is not
   tight — the absolute route really does cost both segments). Both bounds and both effect
   sizes, in one panel.

C  The excess against the out-of-plane twist, binned by theta_rel, with fact 3's closed form
   overlaid as dashed curves. Not decoration: the curves are drawn from
   2 acos(cos(theta_rel/2) cos(psi/2)) - theta_rel with no fitted parameter, so agreement is a
   test of the geometry rather than a summary of the data. It is also the panel that shows why
   B's theta_comp curve hugs the diagonal — at the |psi| this data produces, the closed form
   predicts an excess of hundredths of a degree.

D  Direction angles by joint, MAGNETOMETER. The distortion mechanism.
E  Direction angles by joint, ACCELEROMETER after projection to the joint center. The
   rigid-body mechanism. Same theorem, unrelated physics, which is why both are shown.
F  ECDF of the three direction angles, both fields pooled. Where D and E give the by-joint
   spread, this gives the distribution, including the tails the boxes cut off.

G  The irreducible floors by joint — half the inter-field-angle mismatch — for the projected
   and unprojected accelerometer. The headline comparison, and the only one in the figure that
   no orientation estimate can improve on. The unprojected pair is the control: it shows the
   joint-center projection is what makes the relative floor small, rather than the relative
   pairing alone.

H  The same comparison per subject x joint cell, against the identity line, with the blocked
   significance test behind it. Every point above the line is a cell where the absolute route's
   irreducible error is larger. This is the panel the paper's p-value comes from — the test
   blocks on subject x joint (see plotting/utils.py's DEFAULT_BLOCK_COLS), because a test over
   8M samples from 22 trials would be measuring the sample rate.

I  One window of one trial at full rate, with the driving linear acceleration underneath. The
   only panel that shows the quantities as signals rather than as distributions: the two absolute
   angles rise and fall together with each stride, which is what "common mode" means.

   Read its level, not its smoothness. The window is the BUSIEST one in the trial, chosen that
   way so it cannot be cherry-picked, and the relative angle spikes there too — the joint-center
   projection is least accurate exactly when the segment accelerates hardest, so this is its worst
   window rather than its best. The window medians are drawn on the panel because the level is
   what the panel can actually support.

Rows 2-3 pool every subject, activity and sample on disk. A is analytic. I is one trial, chosen
by --subject/--activity, with its window picked deterministically (see select_dynamic_window).

Each panel is also written as its own file at full size, so one can be dropped into the
manuscript without re-plotting; --composed-only skips those. A three-panel headline figure for
the main text is written alongside the full one.
"""
import argparse
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paths
from plotting import utils as plot_utils  # noqa: F401  (applies the shared paper rcParams on import)
from plotting.sensor_distributions import joint_label
from experiments.experiment_utils import JOINTS
from experiments.relative_vs_absolute import (EXAMPLE_JOINT, FIELD_LABELS, INVARIANT_FIELDS,
                                              SAMPLE_STRIDE, load_trial_table)

PLOTS_DIR = paths.plots_dir("relative_vs_absolute")

EXAMPLE_SUBJECT = 'Subject01'
EXAMPLE_ACTIVITY = 'walking'
EXAMPLE_FIELD = 'acc'  # the time-series panel; the accelerometer is where the swing is visible
WINDOW_S = 4.0  # ~3 gait cycles: long enough to read as periodic, short enough to resolve peaks

# One colour per CORRECTION ROUTE, not per field, and the same colours in all nine panels. The
# figure's whole point is a comparison between routes, so a colour has to mean a route
# everywhere; fields are distinguished by line style and by which panel they are in.
ROUTE_COLORS = {
    'relative': '#1f6f8b',   # the relative residual: what MAJIC uses
    'absolute': '#d1603d',   # per-sensor correction to a global reference, in the joint angle
    'total': '#8c3b22',      # the same, summed over both segments
    'oracle': '#8a8a8a',     # min over the two absolute corrections; not implementable
    'reference': '#4a4a4a',  # the global reference itself, in the schematic
}
# Shared with plotting/acceleration_projection.py, deliberately: both supplementary figures
# compare "what the filter does" against "the naive alternative", and a reader looking at them
# together should not have to relearn which colour is which.

DIRECTION_SERIES = {
    'theta_rel': ('Relative: $\\theta_{rel}$', 'relative'),
    'theta_comp': ('Absolute, joint angle: $\\theta_{comp}$', 'absolute'),
    'theta_sum': ('Absolute, both segments: $\\theta_{J}+\\theta_{K}$', 'total'),
    'theta_min_abs': ('Oracle: $\\min(\\theta_J,\\theta_K)$', 'oracle'),
}
INVARIANT_SERIES = {
    'floor_rel': ('Relative: $|\\beta_J-\\beta_K|/2$', 'relative'),
    'floor_abs': ('Absolute: $\\max|\\beta-\\beta_G|/2$', 'absolute'),
}

# Joints grouped by anatomical level, then side, rather than in JOINTS' order (which walks down
# the right leg and then down the left). Grouping by level is what makes "error grows distally"
# readable, and puts each joint next to its own mirror image so a left/right asymmetry shows up
# as a sanity check.
LEVEL_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']
JOINT_LABELS = {joint: joint_label(joint) for joint in JOINTS}
JOINT_ORDER = [JOINT_LABELS[j] for level in LEVEL_ORDER for j in JOINTS
               if JOINT_LABELS[j].split()[0] == level]

# Collapses left/right into one joint type, matching plotting/paper_figures.py's RENAME_JOINTS.
# Required by the significance testing: the two sides of a joint correlate at ICC ~0.5, so
# blocking on them separately would overstate precision by ~1.5x (see plotting/utils.py).
RENAME_JOINTS = {'R_Hip': 'Hip', 'L_Hip': 'Hip', 'R_Knee': 'Knee', 'L_Knee': 'Knee',
                 'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle', 'Lumbar': 'Lumbar'}

# Whisker percentiles for the by-joint boxes. Not 1.5x IQR: every metric here is a non-negative
# angle with a heavy tail (distortion events for mag, footfalls for acc), so Tukey whiskers
# would sit far inside a huge flier cloud and the box would claim a range the data does not
# have. p5/p95 is stated on the axis instead of implied.
WHISKER_PCT = (5, 95)

# Bin edges for panels B and C. Geometric, because both axes span three orders of magnitude and
# uniform bins would put nearly every sample in the first one.
THETA_REL_BINS = np.geomspace(0.3, 90.0, 16)
PSI_BINS = np.geomspace(0.01, 30.0, 18)
# theta_rel bands panel C draws a separate measured curve and analytic prediction for. Chosen to
# straddle the data's own median (~5-7 deg) so the closed form is tested above and below it.
THETA_REL_BANDS = ((1.0, 3.0), (3.0, 8.0), (8.0, 20.0), (20.0, 60.0))

ECDF_DRAW_POINTS = 2000
RNG_SEED = 0

# ==============================================================================
# Loading
# ==============================================================================

def load_samples(table: str, columns: Sequence[str]) -> pd.DataFrame:
    """Pooled per-sample table, restricted to the columns a panel needs.

    The column list is not a micro-optimization: these tables are millions of rows across every
    trial, and a two-column panel that loaded all of them would dominate this script's memory."""
    df = load_trial_table(table, columns=list(columns))
    if not df.empty and 'joint' in df.columns:
        df['joint_label'] = df['joint'].map(JOINT_LABELS).astype('category')
    return df


def describe_coverage(samples: pd.DataFrame) -> str:
    """One line naming what the pooled panels are pooling, for the figure footer. A
    supplementary figure that does not say how much data is behind it is not checkable."""
    if samples.empty:
        return "no samples on disk"
    trials = samples[['subject', 'activity']].drop_duplicates()
    return (f"{len(samples):,} samples (every {SAMPLE_STRIDE}th) from {len(trials)} trials, "
            f"{trials['subject'].nunique()} subjects, {samples['joint'].nunique()} joints; "
            f"ground-truth orientations from mocap, no filter run")


def label_of(column: str) -> str:
    return {**DIRECTION_SERIES, **INVARIANT_SERIES}[column][0]


def color_of(column: str) -> str:
    return ROUTE_COLORS[{**DIRECTION_SERIES, **INVARIANT_SERIES}[column][1]]

# ==============================================================================
# Panel A: the spherical triangle
# ==============================================================================
# Drawn in orthographic projection rather than with mplot3d. Two reasons: mplot3d cannot
# z-order a filled patch against a line reliably, so the shaded triangle ends up either on top
# of or underneath everything; and an orthographic projection of a sphere is exactly a circle,
# which makes the outline crisp instead of a polygon approximation.

VIEW_RIGHT = np.array([1.0, 0.0, 0.0])
VIEW_UP = np.array([0.0, 1.0, 0.0])
VIEW_OUT = np.cross(VIEW_RIGHT, VIEW_UP)  # toward the viewer; front hemisphere is @ VIEW_OUT > 0

# The three directions, as (polar angle from the view axis, azimuth in the view plane) in
# degrees. Specified this way rather than as raw vectors because what makes the panel legible is
# that all three sit well OFF the view axis and well apart in azimuth: placed near the axis they
# project close to the centre of the disc, the arcs between them come out nearly straight, and
# the whole thing reads as a cone with an apex rather than as a triangle on a surface.
#
# NOT to scale, and the panel says so. A to-scale version would put all three within a few
# degrees of each other, which is the content of panel C.
# Equal polar angles and azimuths 120 deg apart, i.e. an equilateral spherical triangle centred
# on the view axis. Tried asymmetric placements first and they read as a wine glass — two arcs
# converging on a low point — rather than as a patch of surface, which is the one thing this
# panel has to communicate.
SCHEMATIC = {
    'u_parent': (42.0, 150.0),
    'u_child': (42.0, 30.0),
    'reference': (42.0, 270.0),
}

# Tilt of the wireframe's own pole away from the view axis. The wireframe exists only to make the
# disc read as a sphere, and it only does that if its parallels project as visibly curved
# ellipses — with the pole pointed at the viewer they project as concentric circles, which reads
# as a target rather than a globe.
WIREFRAME_TILT_DEG = 68.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _from_view_angles(polar_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit vector at `polar_deg` from the view axis, `azimuth_deg` around it."""
    polar, azimuth = np.radians(polar_deg), np.radians(azimuth_deg)
    return _unit(np.sin(polar) * (np.cos(azimuth) * VIEW_RIGHT + np.sin(azimuth) * VIEW_UP)
                 + np.cos(polar) * VIEW_OUT)


def _project(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthographic projection onto the view plane."""
    points = np.atleast_2d(points)
    return points @ VIEW_RIGHT, points @ VIEW_UP


def _front(points: np.ndarray) -> np.ndarray:
    """Keeps only the points on the hemisphere facing the viewer.

    Without this the far half of every wireframe circle is drawn over the near half and the
    sphere loses its depth entirely."""
    return points[points @ VIEW_OUT >= -1e-9]


def _draw_wireframe(ax: plt.Axes) -> None:
    """Faint parallels and meridians on the front hemisphere, so the disc reads as a sphere."""
    tilt = np.radians(WIREFRAME_TILT_DEG)
    pole = _unit(np.cos(tilt) * VIEW_OUT + np.sin(tilt) * VIEW_UP)
    e1 = _unit(np.cross(pole, VIEW_RIGHT))
    e2 = np.cross(pole, e1)
    angles = np.linspace(0.0, 2 * np.pi, 400)[:, None]

    for latitude in np.radians([-60.0, -30.0, 0.0, 30.0, 60.0]):
        ring = (np.sin(latitude) * pole
                + np.cos(latitude) * (np.cos(angles) * e1 + np.sin(angles) * e2))
        x, y = _project(_front(ring))
        ax.plot(x, y, color='#dcdcdc', linewidth=0.7, zorder=1)
    for longitude in np.radians(np.arange(0.0, 180.0, 30.0)):
        axis = np.cos(longitude) * e1 + np.sin(longitude) * e2
        circle = np.cos(angles) * axis + np.sin(angles) * pole
        x, y = _project(_front(circle))
        ax.plot(x, y, color='#dcdcdc', linewidth=0.7, zorder=1)


def _great_circle_arc(a: np.ndarray, b: np.ndarray, n: int = 120) -> np.ndarray:
    """The minor great-circle arc from unit `a` to unit `b`, as (n, 3) points.

    Slerp rather than linear interpolation followed by normalization: the two agree at the
    endpoints but not in between, and the midpoint of the arc is exactly where the angle labels
    are placed."""
    omega = np.arccos(np.clip(float(a @ b), -1.0, 1.0))
    if omega < 1e-9:
        return np.tile(a, (n, 1))
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)


def _full_great_circle(a: np.ndarray, b: np.ndarray, n: int = 400) -> np.ndarray:
    """The complete great circle through unit `a` and unit `b`. Drawn dashed in the schematic
    so that psi's meaning is visible as a distance: the reference's offset FROM THIS PLANE is
    what the shaded area measures, and fact 4 says the penalty vanishes when it lies on it."""
    normal = _unit(np.cross(a, b))
    e1 = a
    e2 = _unit(np.cross(normal, e1))
    angles = np.linspace(0.0, 2 * np.pi, n)[:, None]
    return np.cos(angles) * e1 + np.sin(angles) * e2


def panel_schematic(ax: plt.Axes) -> None:
    """The spherical triangle (u_J, m_G, u_K), its three geodesics, and its area."""
    u_parent = _from_view_angles(*SCHEMATIC['u_parent'])
    u_child = _from_view_angles(*SCHEMATIC['u_child'])
    reference = _from_view_angles(*SCHEMATIC['reference'])

    ax.add_artist(plt.Circle((0, 0), 1.0, facecolor='#fafafa', edgecolor='#bfbfbf',
                             linewidth=1.6, zorder=0))
    _draw_wireframe(ax)

    # The great circle through the two measurements, extended. Where the reference sits relative
    # to this dashed circle IS the content of fact 4: on it, the penalty is exactly zero.
    x, y = _project(_front(_full_great_circle(u_parent, u_child)))
    ax.plot(x, y, color=ROUTE_COLORS['relative'], linewidth=1.1, linestyle=(0, (5, 4)),
            alpha=0.55, zorder=2)

    # The triangle, filled: its area is psi.
    boundary = np.vstack([_great_circle_arc(u_parent, reference),
                          _great_circle_arc(reference, u_child),
                          _great_circle_arc(u_child, u_parent)])
    bx, by = _project(boundary)
    ax.fill(bx, by, facecolor=ROUTE_COLORS['absolute'], alpha=0.16, linewidth=0, zorder=3)

    # Label radii push each arc's label off the arc: outward for theta_rel (which runs along the
    # top of the triangle) and inward for the two absolute arcs, so none of the three lands on
    # the psi annotation at the centroid.
    arcs = [
        (u_parent, u_child, 'relative', 3.6, '$\\theta_{rel}$', 1.16),
        (u_parent, reference, 'absolute', 2.2, '$\\theta_{J}$', 1.14),
        (u_child, reference, 'absolute', 2.2, '$\\theta_{K}$', 1.14),
    ]
    for start, end, route, width, label, label_radius in arcs:
        arc = _great_circle_arc(start, end)
        x, y = _project(arc)
        ax.plot(x, y, color=ROUTE_COLORS[route], linewidth=width, solid_capstyle='round',
                zorder=5)
        mx, my = _project(arc[len(arc) // 2] * label_radius)
        ax.text(float(mx[0]), float(my[0]), label, color=ROUTE_COLORS[route], fontsize=16,
                fontweight='bold', ha='center', va='center', zorder=7,
                bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 1.5})

    points = [(u_parent, '$\\mathbf{u}_J$', 'relative', 1.20),
              (u_child, '$\\mathbf{u}_K$', 'relative', 1.20),
              (reference, '$\\mathbf{m}_G$', 'reference', 1.22)]
    for direction, label, route, label_radius in points:
        x, y = _project(direction)
        ax.scatter(x, y, s=110, color=ROUTE_COLORS[route], zorder=6, edgecolor='white',
                   linewidth=1.4)
        lx, ly = _project(direction * label_radius)
        ax.text(float(lx[0]), float(ly[0]), label, fontsize=18, fontweight='bold',
                color=ROUTE_COLORS[route], ha='center', va='center', zorder=7)

    # psi, inside the triangle but pushed below its centroid: for an equilateral configuration
    # the centroid projects to the view axis, which is exactly where the dashed great circle
    # crosses, so the label lands on the line.
    cx, cy = _project(_unit(u_parent + u_child + reference))
    ax.text(float(cx[0]), float(cy[0]) - 0.22, '$\\psi$', fontsize=20, fontweight='bold',
            color=ROUTE_COLORS['absolute'], ha='center', va='center', zorder=7)

    ax.text(0.5, -0.03,
            '$\\cos(\\theta_{comp}/2) = \\cos(\\theta_{rel}/2)\\,\\cos(\\psi/2)$'
            '$\\;\\Rightarrow\\;\\theta_{rel} \\leq \\theta_{comp}$',
            transform=ax.transAxes, ha='center', va='top', fontsize=15)
    ax.text(0.5, -0.11,
            '$\\psi$ is the triangle\'s area, and vanishes when $\\mathbf{m}_G$ lies on the '
            'dashed circle',
            transform=ax.transAxes, ha='center', va='top', fontsize=11, color='#555555')
    # No panel cross-reference in this note: the schematic appears in the nine-panel figure, the
    # three-panel headline figure and its own standalone file, and "panel C" means a different
    # panel in each of them.
    ax.text(0.5, -0.18, 'angles exaggerated; measured triangles are near-degenerate slivers',
            transform=ax.transAxes, ha='center', va='top', fontsize=10, fontstyle='italic',
            color='#888888')

    # Asymmetric y limits: the labels need headroom above the sphere but the captions live below
    # the axes, so a symmetric range just adds a dead band under the disc.
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.12, 1.32)
    ax.set_aspect('equal')
    ax.axis('off')

# ==============================================================================
# Panel B: the bounds against the identity line
# ==============================================================================

def panel_identity(ax: plt.Axes, samples: pd.DataFrame,
                   fields: Sequence[str] = ('mag', 'acc')) -> None:
    """theta_comp and theta_sum against theta_rel, with the identity line.

    Median-in-bins with IQR ribbons rather than a scatter: the two curves differ from each other
    by two orders of magnitude in their distance from the diagonal, and a scatter dense enough
    to show theta_sum's spread would bury theta_comp's hugging of the line entirely.

    Both bounds are exact, so nothing can fall below the diagonal, and the panel is not trying
    to establish that. What it shows is HOW FAR ABOVE each one sits — which is the part that is
    empirical and the part the paper has to be careful about.
    """
    styles = {'mag': '-', 'acc': '--'}
    for field in fields:
        subset = samples[samples['field'] == field]
        if subset.empty:
            continue
        binned = subset.assign(bin=pd.cut(subset['theta_rel'], THETA_REL_BINS))
        grouped = binned.groupby('bin', observed=True)
        centers = grouped['theta_rel'].median()
        for column in ('theta_comp', 'theta_sum'):
            quantiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
            ax.fill_between(centers, quantiles[0.25], quantiles[0.75],
                            color=color_of(column), alpha=0.14, linewidth=0)
            ax.plot(centers, quantiles[0.5], color=color_of(column), linewidth=2.2,
                    linestyle=styles.get(field, '-'), marker='o', markersize=3.5)

    limits = (THETA_REL_BINS[0], THETA_REL_BINS[-1])
    ax.plot(limits, limits, color='#444444', linewidth=1.4, linestyle=':', zorder=1)
    ax.text(limits[1], limits[1], ' equality ', color='#444444', fontsize=11, rotation=45,
            ha='right', va='top', fontstyle='italic')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*limits)
    ax.set_xlabel('Relative correction $\\theta_{rel}$ (deg)')
    ax.set_ylabel('Absolute correction (deg)')

    handles = [Line2D([], [], color=color_of(c), linewidth=2.4, label=label_of(c))
               for c in ('theta_comp', 'theta_sum')]
    handles += [Line2D([], [], color='#666666', linewidth=1.8, linestyle=styles[f],
                       label=FIELD_LABELS[f].split(' (')[0]) for f in fields if f in styles]
    ax.legend(handles=handles, loc='upper left', fontsize=10)
    sns.despine(ax=ax)

# ==============================================================================
# Panel C: the excess against the out-of-plane twist, vs the closed form
# ==============================================================================

def closed_form_excess(theta_rel_deg: float, psi_deg: np.ndarray) -> np.ndarray:
    """theta_comp - theta_rel from fact 3, with no fitted parameter.

    2 acos(cos(theta_rel/2) cos(psi/2)) - theta_rel. Drawn on panel C as the prediction the
    measured medians are tested against, which is only meaningful because nothing in it was
    estimated from the data."""
    half_rel = np.radians(theta_rel_deg) / 2.0
    half_psi = np.radians(np.asarray(psi_deg, dtype=float)) / 2.0
    product = np.clip(np.cos(half_rel) * np.cos(half_psi), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(product)) - theta_rel_deg


def panel_mechanism(ax: plt.Axes, samples: pd.DataFrame, field: str = 'mag') -> None:
    """Measured excess against |psi|, in theta_rel bands, with fact 3's closed form dashed.

    One field at a time, because the closed form depends on theta_rel and pooling fields would
    mix two different theta_rel distributions inside each band. Which field is shown does not
    matter to the geometry — that is the point of it being geometry — and the standalone version
    is written for both so that can be checked rather than taken on trust.
    """
    subset = samples[samples['field'] == field]
    if subset.empty:
        return
    colors = sns.color_palette('viridis', n_colors=len(THETA_REL_BANDS))

    for (low, high), color in zip(THETA_REL_BANDS, colors):
        band = subset[(subset['theta_rel'] >= low) & (subset['theta_rel'] < high)]
        if len(band) < 100:
            continue
        binned = band.assign(bin=pd.cut(band['psi_abs'], PSI_BINS))
        grouped = binned.groupby('bin', observed=True)
        centers = grouped['psi_abs'].median()
        measured = grouped['excess'].median()
        ax.plot(centers, measured, color=color, linewidth=2.4, marker='o', markersize=4,
                label=f'{low:g}–{high:g}°', zorder=3)
        # The prediction, at the band's own median theta_rel. Dashed and black so it cannot be
        # mistaken for another data series.
        predicted = closed_form_excess(float(band['theta_rel'].median()), centers.to_numpy())
        ax.plot(centers, predicted, color='black', linewidth=1.2, linestyle='--', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Out-of-plane twist $|\\psi|$ (deg)')
    ax.set_ylabel('Excess $\\theta_{comp}-\\theta_{rel}$ (deg)')

    handles = ax.get_legend_handles_labels()[0]
    handles.append(Line2D([], [], color='black', linewidth=1.2, linestyle='--',
                          label='Fact 3, no free parameters'))
    ax.legend(handles=handles, loc='upper left', fontsize=10,
              title=f'$\\theta_{{rel}}$ band ({FIELD_LABELS[field].split(" (")[0].lower()})',
              title_fontsize=10)
    sns.despine(ax=ax)

# ==============================================================================
# Panels D, E, G: by-joint distributions
# ==============================================================================

def _box_stats(values: pd.Series) -> Dict[str, float]:
    """5-number summary in matplotlib's bxp format, with p5/p95 whiskers (see WHISKER_PCT)."""
    lo, hi = np.percentile(values, WHISKER_PCT)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {'med': median, 'q1': q1, 'q3': q3, 'whislo': lo, 'whishi': hi, 'fliers': []}


def panel_grouped_boxes(ax: plt.Axes, samples: pd.DataFrame, value_cols: Sequence[str],
                        ylabel: str, log: bool = True, legend_loc: str = 'upper left') -> None:
    """Grouped box plot: one box per (joint, series).

    Drawn with ax.bxp from precomputed percentiles rather than seaborn.boxplot on melted data.
    Two reasons: melting several million rows to draw twenty boxes doubles the memory for
    nothing, and the quantiles are then exactly the ones the summary parquet reports instead of
    whatever seaborn recomputes after its own filtering.
    """
    order = [j for j in JOINT_ORDER if j in set(samples['joint_label'])]
    if not order or not value_cols:
        return
    span = 0.8
    width = span / len(value_cols)
    offsets = (np.arange(len(value_cols)) - (len(value_cols) - 1) / 2.0) * width

    for offset, column in zip(offsets, value_cols):
        stats, positions = [], []
        for index, joint in enumerate(order):
            values = samples.loc[samples['joint_label'] == joint, column].dropna()
            if values.empty:
                continue
            stats.append({'label': joint, **_box_stats(values)})
            positions.append(index + offset)
        if not stats:
            continue
        color = color_of(column)
        ax.bxp(stats, positions=positions, widths=width * 0.85, showfliers=False,
               patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.4},
               boxprops={'facecolor': color, 'edgecolor': color, 'alpha': 0.75},
               whiskerprops={'color': color}, capprops={'color': color})

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[Patch(facecolor=color_of(c), alpha=0.75, label=label_of(c))
                       for c in value_cols], loc=legend_loc, fontsize=10)
    ax.text(0.99, 0.02, f'boxes: IQR, whiskers: p{WHISKER_PCT[0]}–p{WHISKER_PCT[1]}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)


def panel_invariant_boxes(ax: plt.Axes, invariants: pd.DataFrame) -> None:
    """The irreducible floors by joint, projected accelerometer, plus the unprojected control.

    The control is drawn as an outlined box rather than a fourth colour: it is the SAME relative
    quantity measured without the joint-center projection, so it belongs visually with the
    relative box it modifies, not as a separate route.
    """
    primary = invariants[invariants['field'] == INVARIANT_FIELDS[0]]
    if primary.empty:
        return
    panel_grouped_boxes(ax, primary, ('floor_rel', 'floor_abs'),
                        'Irreducible orientation error (deg)', log=True)

    control = invariants[invariants['field'] == INVARIANT_FIELDS[1]]
    order = [j for j in JOINT_ORDER if j in set(primary['joint_label'])]
    if control.empty or not order:
        return
    medians = [control.loc[control['joint_label'] == j, 'floor_rel'].median() for j in order]
    ax.plot(range(len(order)), medians, color=ROUTE_COLORS['relative'], linewidth=0,
            marker='v', markersize=9, markerfacecolor='white',
            markeredgecolor=ROUTE_COLORS['relative'], markeredgewidth=1.8, zorder=6)
    handles = ax.get_legend().legend_handles + [
        Line2D([], [], color=ROUTE_COLORS['relative'], linewidth=0, marker='v', markersize=9,
               markerfacecolor='white', markeredgewidth=1.8,
               label='Relative, no joint-center projection')]
    ax.legend(handles=handles, loc='upper left', fontsize=10)

# ==============================================================================
# Panel F: ECDF
# ==============================================================================

def panel_ecdf(ax: plt.Axes, samples: pd.DataFrame,
               value_cols: Sequence[str] = ('theta_rel', 'theta_comp', 'theta_sum')) -> None:
    """Empirical CDF of the pooled direction angles, one curve per route.

    Computed on every sample and thinned only for drawing, so the curve is exact wherever it is
    read.

    theta_rel and theta_comp COINCIDE here — that is fact 2 being tight, and it is the panel's
    main content, so the two are drawn to make the coincidence visible rather than to hide one
    curve under the other: theta_rel goes down thick and pale underneath, theta_comp over it as a
    thin dashed line. Drawn as two equal-weight solid lines (the first version of this panel) the
    relative curve is invisible and the panel silently shows two series while its legend claims
    three.
    """
    widths = {'theta_rel': 5.0}
    styles = {'theta_comp': (0, (5, 3))}
    for column in value_cols:
        values = np.sort(samples[column].dropna().to_numpy())
        if not len(values):
            continue
        fraction = np.arange(1, len(values) + 1) / len(values)
        step = max(len(values) // ECDF_DRAW_POINTS, 1)
        ax.plot(values[::step], fraction[::step], color=color_of(column),
                linewidth=widths.get(column, 2.0), linestyle=styles.get(column, '-'),
                alpha=0.75 if column in widths else 1.0,
                label=f'{label_of(column)}  (median {np.median(values):.2f}°)')
        ax.vlines(np.median(values), 0, 0.5, color=color_of(column), linewidth=1.0,
                  linestyle=':', alpha=0.8)
    ax.axhline(0.5, color='#999999', linewidth=0.8, linestyle=':')
    ax.set_xscale('log')
    ax.set_xlabel('Correction angle (deg)')
    ax.set_ylabel('Fraction of samples below')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)

# ==============================================================================
# Panel H: paired cells, with the blocked significance test
# ==============================================================================

# The three arms the significance test compares, as (label, invariant field, column). The
# unprojected relative arm is here for two reasons. It is scientifically the right control — it
# separates "pair the two sensors" from "pair them at the same physical point" — and it is what
# makes the omnibus test valid: plot_utils.test_panels leads with a Friedman, which needs at
# least three groups, so a bare two-arm comparison comes back as "Friedman failed" and no
# pairwise p-value is ever corrected or reported.
TEST_ARMS = (
    (INVARIANT_SERIES['floor_rel'][0], INVARIANT_FIELDS[0], 'floor_rel'),
    ('Relative, unprojected', INVARIANT_FIELDS[1], 'floor_rel'),
    (INVARIANT_SERIES['floor_abs'][0], INVARIANT_FIELDS[0], 'floor_abs'),
)
TEST_ARM_ORDER = [label for label, _, _ in TEST_ARMS]


def block_table(invariant_stats: pd.DataFrame) -> pd.DataFrame:
    """The per-cell table the significance test blocks on: one row per subject x activity x
    joint x arm, carrying that cell's median floor.

    Built from invariant_stats (computed over every sample of the trial) rather than from the
    decimated per-sample table, and reduced to a median per cell rather than fed as samples.
    Both choices matter: a Wilcoxon over millions of samples from 19 trials measures the sample
    rate, not the effect, because consecutive samples at 100 Hz are almost perfectly correlated.
    plot_utils then blocks one level coarser still, averaging over side and activity.
    """
    if invariant_stats.empty:
        return pd.DataFrame()
    frames = []
    for label, field, column in TEST_ARMS:
        cells = invariant_stats[invariant_stats['field'] == field]
        if cells.empty:
            continue
        frames.append(pd.DataFrame({
            'subject': cells['subject'].astype(str),
            'activity': cells['activity'].astype(str),
            'joint_name': cells['joint'].astype(str).replace(RENAME_JOINTS),
            'route': label,
            'floor_deg': cells[f'{column}_p50'].to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def panel_paired_cells(ax: plt.Axes, invariant_stats: pd.DataFrame) -> Dict[str, object]:
    """Per-cell relative floor against absolute floor, with the identity line, one marker per
    joint level. Returns the test result so the caller can put it in the epilog.

    Unlike panels B and F this comparison is NOT bounded — fact 6 bounds dip_rel by dip_sum, not
    by max(dip_parent, dip_child) — so points below the line are possible and their count is the
    honest measure of how often the relative route actually wins. That count is annotated.
    """
    blocks = block_table(invariant_stats)
    if blocks.empty:
        return {}
    wide = blocks.pivot_table(index=['subject', 'activity', 'joint_name'], columns='route',
                              values='floor_deg').dropna()
    relative_label = INVARIANT_SERIES['floor_rel'][0]
    absolute_label = INVARIANT_SERIES['floor_abs'][0]
    if relative_label not in wide.columns or absolute_label not in wide.columns:
        return {}

    levels = [level for level in LEVEL_ORDER
              if level in set(wide.index.get_level_values('joint_name'))]
    markers = dict(zip(levels, ('o', 's', '^', 'D', 'v')))
    palette = dict(zip(levels, sns.color_palette('crest', n_colors=max(len(levels), 1))))
    for level in levels:
        subset = wide[wide.index.get_level_values('joint_name') == level]
        ax.scatter(subset[relative_label], subset[absolute_label], s=58,
                   marker=markers[level], color=palette[level], edgecolor='white',
                   linewidth=0.8, alpha=0.9, label=level, zorder=3)

    low = float(min(wide[relative_label].min(), wide[absolute_label].min())) * 0.7
    high = float(max(wide[relative_label].max(), wide[absolute_label].max())) * 1.4
    ax.plot([low, high], [low, high], color='#444444', linewidth=1.4, linestyle=':', zorder=1)
    ax.fill_between([low, high], [low, high], high, color=ROUTE_COLORS['relative'],
                    alpha=0.06, linewidth=0, zorder=0)

    above = int((wide[absolute_label] > wide[relative_label]).sum())
    ax.text(0.03, 0.97, f'relative smaller in {above}/{len(wide)} cells\n'
                        f'median ratio {float((wide[absolute_label] / wide[relative_label]).median()):.2f}×',
            transform=ax.transAxes, ha='left', va='top', fontsize=11, fontweight='bold',
            color=ROUTE_COLORS['relative'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect('equal')
    ax.set_xlabel('Relative route, irreducible error (deg)')
    ax.set_ylabel('Absolute route, irreducible error (deg)')
    ax.legend(loc='lower right', fontsize=10, title='Joint', title_fontsize=10)
    sns.despine(ax=ax)

    return plot_utils.test_panels({'Irreducible error': blocks}, 'floor_deg', 'route',
                                  TEST_ARM_ORDER)

# ==============================================================================
# Panel I: time series
# ==============================================================================

def select_dynamic_window(trace: pd.DataFrame, window_s: float = WINDOW_S
                          ) -> Tuple[float, float]:
    """Picks the `window_s` window whose driving signal moves the most, as (start, end).

    Deterministic and stated rather than hand-picked, because the choice of window decides what
    the panel appears to show: a quiet window puts all three angles near zero and would suggest
    there is nothing to see, while the most dynamic window is where the absolute route's error is
    largest and where the relative route can most easily be seen to fail. Scored by the rolling
    standard deviation of `drive` summed over the window — the busiest stretch, not the
    highest-amplitude one, so a single heel-strike spike does not win it.
    """
    timestamps = trace['timestamp'].to_numpy()
    if len(timestamps) < 3:
        return (0.0, 1.0)
    fs = 1.0 / np.median(np.diff(timestamps))
    width = max(int(round(window_s * fs)), 2)
    activity = trace['drive'].rolling(window=max(width // 8, 2)).std()
    score = activity.rolling(window=width).sum().to_numpy()
    if np.all(np.isnan(score)):
        return float(timestamps[0]), float(timestamps[min(width, len(timestamps) - 1)])
    end = int(np.nanargmax(score))
    return float(timestamps[max(end - width, 0)]), float(timestamps[end])


def panel_timeseries(ax: plt.Axes, traces: pd.DataFrame, field: str = EXAMPLE_FIELD,
                     window: Optional[Tuple[float, float]] = None) -> None:
    """One window at full rate, with the driving linear acceleration shaded underneath on a
    second axis.

    The two absolute angles are drawn separately rather than as theta_sum, because the point is
    that they move TOGETHER — they are the common mode — and a sum would hide that by
    construction.

    WHAT THIS PANEL DOES AND DOES NOT SHOW. select_dynamic_window picks the BUSIEST window of the
    trial, deliberately, so that the window is not cherry-picked to flatter the method. The
    consequence is that the relative angle spikes here too: the joint-center projection is least
    accurate exactly when the segment is accelerating hardest, so this is the worst window for it
    rather than the best. What survives is the LEVEL — the relative angle spends most of the
    window well below both absolute ones — and that is what the window medians drawn on the panel
    quote. A panel captioned as though the relative angle stayed flat would be describing a
    quieter window than this one.
    """
    trace = traces[traces['field'] == field].sort_values('timestamp')
    if trace.empty:
        return
    if window is None:
        window = select_dynamic_window(trace)
    t0, t1 = window
    trace = trace[(trace['timestamp'] >= t0) & (trace['timestamp'] <= t1)]
    if trace.empty:
        return
    rel_time = trace['timestamp'].to_numpy() - t0

    drive_ax = ax.twinx()
    drive_ax.fill_between(rel_time, 0, trace['drive'].to_numpy(), color='#cccccc', alpha=0.55,
                          linewidth=0, zorder=0)
    drive_ax.set_ylabel('$|a_{world} - g|$ (m/s²)', color='#7a7a7a', fontsize=13)
    drive_ax.tick_params(axis='y', colors='#7a7a7a')
    drive_ax.set_ylim(0, float(trace['drive'].max()) * 3.0)  # keep the shading in the lower third
    drive_ax.grid(False)
    drive_ax.spines[['top', 'left']].set_visible(False)

    for column, style, width, label in (
            ('theta_parent', '-', 1.5, 'Absolute, parent $\\theta_J$'),
            ('theta_child', '--', 1.5, 'Absolute, child $\\theta_K$'),
            ('theta_rel', '-', 2.6, 'Relative $\\theta_{rel}$')):
        color = ROUTE_COLORS['absolute'] if column != 'theta_rel' else ROUTE_COLORS['relative']
        values = trace[column].to_numpy()
        ax.plot(rel_time, values, color=color, linewidth=width, linestyle=style, label=label,
                zorder=3, solid_capstyle='round')
        # Window medians, because the level is the claim this panel can actually support (see the
        # docstring) and reading a level off three overlapping oscillating traces is guesswork.
        if column != 'theta_child':
            median = float(np.median(values))
            ax.axhline(median, color=color, linewidth=1.0, linestyle=':', alpha=0.9, zorder=2)
            ax.text(1.0, median, f' median {median:.1f}°', transform=ax.get_yaxis_transform(),
                    color=color, fontsize=10, fontweight='bold', ha='right',
                    va='bottom' if column == 'theta_parent' else 'top', zorder=8,
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none', 'pad': 1.0})

    ax.set_zorder(drive_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlim(0, float(rel_time[-1]) if len(rel_time) else 1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Correction angle (deg)')
    ax.legend(loc='upper left', fontsize=10, ncol=1)
    sns.despine(ax=ax, right=False)

# ==============================================================================
# Composed figures
# ==============================================================================

def _label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=20, fontweight='bold',
            va='bottom', ha='left')


def figure_supplement(samples: pd.DataFrame, invariants: pd.DataFrame,
                      invariant_stats: pd.DataFrame, traces: pd.DataFrame,
                      subject: str, activity: str,
                      filename: str = 'relative_vs_absolute.png',
                      save: bool = True, show: bool = False) -> Dict[str, object]:
    """The full nine-panel figure. Returns the significance results from panel H.

    Not routed through plot_utils.finalize_and_save_plot: that helper runs tight_layout over the
    whole figure, and this layout needs a constrained layout to keep panel H's square aspect and
    panel I's twin axis from colliding with their neighbours.
    """
    fig = plt.figure(figsize=(20, 18), layout='constrained')
    axes = fig.subplots(3, 3)

    results: Dict[str, object] = {}
    panels = [
        (axes[0][0], 'A', 'The geometry', lambda ax: panel_schematic(ax)),
        (axes[0][1], 'B', 'Both bounds, and their sizes',
         lambda ax: panel_identity(ax, samples)),
        (axes[0][2], 'C', 'Why the joint-angle bound is tight',
         lambda ax: panel_mechanism(ax, samples, 'mag')),
        (axes[1][0], 'D', 'Magnetometer, by joint',
         lambda ax: panel_grouped_boxes(ax, samples[samples['field'] == 'mag'],
                                       ('theta_rel', 'theta_comp', 'theta_sum'),
                                       'Correction angle (deg)')),
        (axes[1][1], 'E', 'Accelerometer, by joint',
         lambda ax: panel_grouped_boxes(ax, samples[samples['field'] == 'acc'],
                                       ('theta_rel', 'theta_comp', 'theta_sum'),
                                       'Correction angle (deg)')),
        (axes[1][2], 'F', 'Pooled distribution',
         lambda ax: panel_ecdf(ax, samples)),
        (axes[2][0], 'G', 'Irreducible error, by joint',
         lambda ax: panel_invariant_boxes(ax, invariants)),
        (axes[2][1], 'H', 'Irreducible error, per subject × joint',
         lambda ax: results.update(panel_paired_cells(ax, invariant_stats))),
        (axes[2][2], 'I', 'One stride cycle, full rate',
         lambda ax: panel_timeseries(ax, traces)),
    ]
    for ax, letter, title, draw in panels:
        draw(ax)
        ax.set_title(title, fontsize=15)
        _label_panel(ax, letter)

    fig.suptitle('Relative versus absolute correction of a vector field across a joint\n'
                 + describe_coverage(samples), fontsize=22, fontweight='bold')
    _save(fig, filename, save, show)
    return results


def figure_headline(samples: pd.DataFrame, invariants: pd.DataFrame,
                    invariant_stats: pd.DataFrame,
                    filename: str = 'relative_vs_absolute_headline.png',
                    save: bool = True, show: bool = False) -> None:
    """Three panels for the main text: the geometry, the bounds, and the irreducible floors.

    Deliberately not a crop of the supplement. A main-text figure has to carry the claim on its
    own, so it is the schematic (what is being claimed), panel B (both bounds and the fact that
    one of them is tight) and panel H (the effect that survives, per replicate)."""
    fig = plt.figure(figsize=(20, 6.4), layout='constrained')
    axes = fig.subplots(1, 3)
    for ax, letter, title, draw in (
            (axes[0], 'A', 'The geometry', lambda ax: panel_schematic(ax)),
            (axes[1], 'B', 'Both bounds, and their sizes',
             lambda ax: panel_identity(ax, samples)),
            (axes[2], 'C', 'Irreducible error, per subject × joint',
             lambda ax: panel_paired_cells(ax, invariant_stats))):
        draw(ax)
        ax.set_title(title, fontsize=15)
        _label_panel(ax, letter)
    fig.suptitle('A relative correction is bounded below the absolute one, and its irreducible '
                 'part is half the size', fontsize=20, fontweight='bold')
    _save(fig, filename, save, show)


def _save(fig: plt.Figure, filename: str, save: bool, show: bool) -> None:
    if save:
        path = paths.ensure_parent(PLOTS_DIR / filename)
        fig.savefig(path, dpi=400, bbox_inches='tight')
        print(f"Saved plot to {path}")
    if show:
        plt.show()
    plt.close(fig)

# ==============================================================================
# Standalone panels
# ==============================================================================

def _standalone(draw, filename: str, title: str, figsize: Tuple[float, float],
                samples: Optional[pd.DataFrame] = None, save: bool = True,
                show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    epilog = describe_coverage(samples) if samples is not None else None
    plot_utils.finalize_and_save_plot(fig, title, filename, PLOTS_DIR, epilog=epilog,
                                     save=save, show=show)


def figures_standalone(samples: pd.DataFrame, invariants: pd.DataFrame,
                       invariant_stats: pd.DataFrame, traces: pd.DataFrame,
                       subject: str, activity: str, save: bool = True,
                       show: bool = False) -> None:
    """Every panel again at full size, one file each, for the manuscript and for reading the
    ones that are dense at composed size (B, C and H all are)."""
    _standalone(panel_schematic, 'schematic.png',
                'The spherical triangle behind the relative correction', (8, 8),
                None, save, show)
    _standalone(lambda ax: panel_identity(ax, samples), 'bounds.png',
                'Both bounds against the identity line', (9, 7), samples, save, show)
    for field in ('mag', 'acc'):
        _standalone(lambda ax, f=field: panel_mechanism(ax, samples, f),
                    f'mechanism_{field}.png',
                    f'Excess vs out-of-plane twist against the closed form '
                    f'({FIELD_LABELS[field]})', (9, 7), samples, save, show)
        _standalone(lambda ax, f=field: panel_grouped_boxes(
            ax, samples[samples['field'] == f],
            ('theta_rel', 'theta_comp', 'theta_sum', 'theta_min_abs'),
            'Correction angle (deg)'), f'by_joint_{field}.png',
            f'Correction angles by joint ({FIELD_LABELS[field]})', (12, 6.5),
            samples, save, show)
    _standalone(lambda ax: panel_ecdf(ax, samples), 'ecdf.png',
                'Pooled distribution of the correction angles', (9, 6.5), samples, save, show)
    _standalone(lambda ax: panel_invariant_boxes(ax, invariants), 'irreducible_by_joint.png',
                'Irreducible orientation error by joint', (12, 6.5), invariants, save, show)
    _standalone(lambda ax: panel_paired_cells(ax, invariant_stats), 'irreducible_paired.png',
                'Irreducible orientation error per subject × joint', (8.5, 8), None, save, show)
    _standalone(lambda ax: panel_timeseries(ax, traces),
                'timeseries.png',
                f'Absolute and relative correction angles at the '
                f'{JOINT_LABELS.get(EXAMPLE_JOINT, EXAMPLE_JOINT)}, busiest window '
                f'({subject}, {activity})', (12, 6), None, save, show)

# ==============================================================================
# CLI
# ==============================================================================

# Columns the pooled panels need, listed once so the load reads exactly this set and a panel
# that needs a new column fails loudly here rather than silently loading everything.
DIRECTION_COLUMNS = ['joint', 'field', 'theta_rel', 'theta_comp', 'theta_sum', 'theta_min_abs',
                     'psi_abs', 'excess']
INVARIANT_COLUMNS = ['joint', 'field', 'floor_rel', 'floor_abs', 'dip_rel', 'dip_sum']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subject', default=EXAMPLE_SUBJECT,
                        help="Trial shown in the time-series panel, e.g. Subject01.")
    parser.add_argument('--activity', default=EXAMPLE_ACTIVITY)
    parser.add_argument('--joint', default=EXAMPLE_JOINT,
                        help=f"Joint for the time-series panel. Only joints the experiment "
                             f"stores traces for are available (default {EXAMPLE_JOINT}).")
    parser.add_argument('--composed-only', action='store_true',
                        help="Write only the composed figures, not the per-panel files.")
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    print("Loading per-trial tables...")
    samples = load_samples('angle_samples', DIRECTION_COLUMNS)
    if samples.empty:
        print("No per-sample table found. Run `python -m experiments.relative_vs_absolute` "
              "first.")
        return
    invariants = load_samples('invariant_samples', INVARIANT_COLUMNS)
    invariant_stats = load_trial_table('invariant_stats')
    print(describe_coverage(samples))

    subject_id = args.subject.replace('Subject', '')
    traces = load_trial_table('traces', subjects=[subject_id], activities=[args.activity])
    traces = traces[traces['joint'] == args.joint] if not traces.empty else traces
    if traces.empty:
        print(f"No traces for {args.subject} {args.activity} {args.joint}; the time-series "
              f"panel will be empty. The experiment stores traces for {EXAMPLE_JOINT} only.")

    results = figure_supplement(samples, invariants, invariant_stats, traces,
                                args.subject, args.activity, show=args.show)
    figure_headline(samples, invariants, invariant_stats, show=args.show)

    if results:
        report = plot_utils.significance_report(results, 'floor_deg')
        plot_utils._emit_significance_report(report, 'relative_vs_absolute.png', PLOTS_DIR)

    if not args.composed_only:
        figures_standalone(samples, invariants, invariant_stats, traces,
                           args.subject, args.activity, show=args.show)
    print(f"\nFigures under {PLOTS_DIR}")


if __name__ == '__main__':
    main()
