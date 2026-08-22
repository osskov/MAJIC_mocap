# Restricting what the magnetometer is allowed to correct

Derivation and measurements for the question: *should the magnetometer only be allowed to correct
in the direction perpendicular to the acceleration at the joint centre?*

**Short answer, in three parts.**

1. "Perpendicular to the acceleration" names **two different modifications** that are easy to
   conflate: a change to the data (project the field) and a change to the measurement function
   (project the residual and the Jacobian). They are not alternatives and neither subsumes the
   other -- §7 shows no data transformation can do what the second one does.
2. On **Al Borno** every version of the restriction beats `mag_on`, by up to 2.6 deg at the ankle,
   and the per-joint gain is predicted by a filter-free diagnostic at r = -0.97.
3. On **IMoVE** it does not replicate at all -- and the same diagnostic says why, which turns the
   result into a precondition you can test on a new dataset before running any filter (§11). That
   precondition, not the arm, is the useful output.

All equations below are written in plain ASCII inside code blocks so they render anywhere.
Notation follows the code (`p` = parent = `J` in the manuscript, `c` = child = `K`).

Reproduce every number here with:

```
python -m scratch.mag_heading_only --dataset alborno        # the mechanism, no filter run
python -m scratch.mag_heading_only --dataset imove
python -m scratch.mag_heading_only_pilot --dataset alborno  # the six filter arms
python -m scratch.mag_heading_only_pilot --dataset imove
python -m scratch.mag_heading_only_report                   # the cross-dataset comparison
```

---

## 1. Setup

The filter's state is the 6-D error state `eta = [eta_p, eta_c]`, applied as right (body-frame)
perturbations of each segment's estimate:

```
R_wp  <-  R_wp exp([eta_p]x)          R_wc  <-  R_wc exp([eta_c]x)
```

A vector sensor enters as a *relative* measurement — the two segments are compared against each
other, with no world reference anywhere:

```
h(eta) = R_wp exp([eta_p]x) v_p  -  R_wc exp([eta_c]x) v_c
```

At `eta = 0` this gives the residual and Jacobian the code builds
(`RelativeFilterPlus.get_h`, `get_H_jacobian`):

```
e = v_wp - v_wc                                   where  v_wp = R_wp v_p
H = [ R_wp [v_p]x^T ,  -R_wc [v_c]x^T ]
```

## 2. The one identity everything follows from

Body-frame perturbations are awkward; world-frame ones are not. Substitute
`delta_p = R_wp eta_p`, `delta_c = R_wc eta_c` and use `R(a x b) = (Ra) x (Rb)`:

```
R_wp exp([eta_p]x) v_p  ~=  R_wp (v_p + eta_p x v_p)
                         =  v_wp + (R_wp eta_p) x (R_wp v_p)
                         =  v_wp + delta_p x v_wp
```

so

```
h  ~=  (v_wp - v_wc)  +  delta_p x v_wp  -  delta_c x v_wc
```

Two things fall out. First, when the two readings agree (`v_wp ~= v_wc ~= v_w`) a *common*
rotation `delta_p = delta_c` cancels exactly — the relative measurement is blind to common-mode
rotation, which is why absolute heading comes from the mocap seed and why only the joint angle is
scored. Second, writing `delta = delta_c - delta_p` for the **relative** rotation error:

```
    dh = -( delta x v_w )                                                       (*)
```

That is the whole story. Two corollaries do all the work below:

- **`dh = 0` if and only if `delta` is parallel to `v_w`.** A vector-matching measurement is
  blind to rotation about its own vector, and to nothing else. Its Jacobian therefore has
  **rank exactly 2**, for every non-zero `v_w`. This is the fact that answers the
  "can I just preprocess the field?" question in §7.
- The image of `delta -> -delta x v_w` is the plane perpendicular to `v_w`, so the residual lives
  in that plane.

### The trap

There are two different 3-spaces in play and `(*)` maps between them:

| space | what lives in it | units |
|---|---|---|
| **rotation space** | `delta`, the relative orientation error | rad |
| **measurement space** | `e`, `h`, the residual | m/s^2, or field units |

The cross product carries one to the other, rotating by 90 degrees and scaling by `|v_w|`. The
same symbol can appear in both with entirely unrelated meanings — `w_hat` below is the
*measurement-space* direction that heading errors show up in, **and** a *rotation-space* tilt
axis. Conflating the two is the easiest way to get this wrong.

## 3. The frame

Let `a_w` be the accelerometer reading projected to the joint centre and `m_w` the field, both in
the world frame, with `a_hat`, `m_hat` their unit vectors and `theta = angle(a_hat, m_hat)`.
Define

```
w_hat = unit( a_hat x m_hat )        perpendicular to both
t_hat = m_hat x w_hat                a_hat with its m_hat component removed, normalised
p_hat = w_hat x a_hat                the tilt direction that is not w_hat
```

This gives two orthonormal frames sharing `w_hat`, related by a rotation of `theta` about it:

```
{ a_hat, p_hat, w_hat }        and        { m_hat, t_hat, w_hat }
```

with (verifiable by putting `m_hat = z`, `a_hat = sin(theta) x + cos(theta) z`):

```
t_hat = sin(theta) a_hat  -  cos(theta) p_hat
m_hat = cos(theta) a_hat  +  sin(theta) p_hat
```

On Al Borno, `theta` runs **133 to 152 degrees**, so `sin(theta) = 0.47..0.73` and
`|cos(theta)| = 0.68..0.88`. Both matter, and the second being the larger of the two is why the
contamination discussed in §6 is not a rounding error.

### Classifying the rotation DOF

Apply `(*)` to the accelerometer, `v_w = a_w`:

- **heading** = `delta` parallel to `a_hat`. The accelerometer's null direction. Note this is the
  *measured* joint-centre acceleration, not gravity; under load the two differ by tens of degrees,
  and the null direction moves with it.
- **tilt** = the plane `span{ p_hat, w_hat }`. The accelerometer constrains both of these, and it
  does so well: block-averaged over 1 s, the two accelerometers' disagreement demands only
  0.7-1.6 degrees of spurious tilt.

The magnetometer's *only* non-redundant job is heading.

## 4. `mag_on`: the shipped arm

`(*)` with `v_w = m_w`: the mag constrains the two DOF perpendicular to `m_hat`, i.e.
`span{ t_hat, w_hat }`. Rank 2. Written out, its two independent constraints are

```
(i)    delta . t_hat  =  sin(theta) (delta . a_hat)  -  cos(theta) (delta . p_hat)
(ii)   delta . w_hat
```

- **(i) is mixed.** It carries heading at weight `sin(theta)` and tilt (about `p_hat`) at weight
  `cos(theta)`.
- **(ii) is pure tilt.** `w_hat . a_hat = 0` exactly, so this constraint has *zero* heading
  content. It is entirely redundant with the accelerometer, and it is the channel through which a
  magnetic bias is injected into tilt.

## 5. Why that matters here: the bias is slow

Measured against mocap rotations, so every residual is pure sensor disagreement
(`scratch/mag_heading_only.py`). The minimum-norm rotation explaining a sensor pair's residual is
`delta = (v_w x e)/|v_w|^2`, which is perpendicular to `v_w` by construction — so for the
accelerometer `delta . a_hat` is identically zero, the formal statement that it cannot demand a
heading correction.

On Al Borno (the IMoVE column of this table is what kills the effect there -- see §11):

| spurious **tilt** demanded | accelerometer | magnetometer |
|---|---|---|
| full bandwidth, RMS | 12-29 deg | 5-23 deg |
| block-averaged over 1 s | **0.7-1.6 deg** | **2.9-11.5 deg** |

At full bandwidth the accelerometer looks *worse*, which would say this whole idea backfires. But
the accelerometer's demand is heel-strike transients and lever-arm residual, which a filter
averages away; the magnetometer's is field non-uniformity, which is quasi-static and which the
filter tracks. In the DOF the two share, the magnetometer holds a **3-7x larger sustained error**.

And it holds it at comparable weight. A vector sensor's authority per DOF it can see goes as
`(|v|/sigma_v)^2`, which the default stds put within 2% nominally
(`9.81/0.09695 = 101` against `1/0.009695 = 103`), and at 0.72-0.86 as measured.

## 6. Modification A: project the residual (`heading_only_sensor`)

By `(*)`, a heading error `delta = q a_hat` produces

```
dh = -q ( a_hat x m_w ) = -q |m_w| sin(theta) w_hat
```

so `w_hat` is the *measurement-space* direction that heading errors appear in. Contract the
residual **and** the Jacobian with it:

```
e  <-  w_hat ( w_hat . e )
H  <-  w_hat ( w_hat^T H )
```

The retained scalar then responds as

```
dh_s = w_hat . ( -delta x m_w ) = -delta . ( m_w x w_hat ) = -|m_w| ( delta . t_hat )
```

**It keeps constraint (i) and drops (ii). Rank 1.**

### What it does not do

It does **not** make the magnetometer's influence on tilt zero. Constraint (i) still carries
`cos(theta)` of tilt about `p_hat`, at *exactly* the sensitivity `mag_on` had. That is irreducible
for a magnetometer: rotation about `m_hat` is invisible to it under any scheme, so `t_hat` is the
only direction with any heading content available, and `t_hat` is mixed. Since
`|cos(theta)| > sin(theta)` over most of this dataset's range, the *retained* row is in fact more
sensitive to `p_hat` tilt than to heading.

It still helps, because the accelerometer pins `p_hat` and `w_hat` tightly, so the posterior
attributes most of that row to `delta . a_hat`. What was removed is the constraint that fought the
accelerometer directly, with a biased value, and offered nothing in exchange.

### Why rank-1 rather than a genuine 1-row measurement

Keeping three rows at rank 1 leaves the measurement layout — and the compiled kernel's `MEAS_DIM`
— untouched. It is not an approximation of the 4-row filter, it *is* the 4-row filter: the two
discarded rows carry residual 0 and Jacobian 0 so they vanish from `n = -Ke` and from `KH`, and
their block of `M R M^T` is `2 sigma^2 I` with no off-diagonal terms, which makes `S` block
diagonal between them and everything else. `S^-1` restricted to the live rows is then exactly what
a 4-row filter would form. Verified against a hand-built 3-acc-rows-plus-one-scalar filter:
identical to 1e-21.

`M` is deliberately **not** projected. Projecting it would zero that block, make `S` singular, and
fail the Cholesky solve every sample. The exactness above needs the per-axis mag stds to be equal,
which is what the pipeline passes; with anisotropic stds the discarded rows couple back in.

## 7. Modification B: project the field (preprocessing)

The other reading of the same English sentence: compute `m_perp = m - a_hat (a_hat . m)` in each
body frame and hand *that* to an unmodified filter. In the frame of §3,

```
m_perp,w = m_w - a_hat ( a_hat . m_w ) = |m_w| sin(theta) p_hat
```

The preprocessed field points along `p_hat`. So by `(*)` with `v_w = m_perp,w` it constrains the
two DOF perpendicular to `p_hat`, namely `span{ a_hat, w_hat }` — **still rank 2**:

```
(i')   delta . a_hat      PURE heading, at the same weight |m_w| sin(theta) as before
(ii')  delta . w_hat      the same pure-tilt constraint, at half the weight
```

Preprocessing **purifies (i) into (i')** and **keeps (ii')**. It rotates the sensor's blind
direction from `m_hat` to `p_hat`; it does not reduce its rank.

### Why no preprocessing can ever reach rank 1

This is the direct answer to "isn't this just preprocessing?". By `(*)`, the Jacobian of *any*
vector-matching update is `delta -> -delta x v_w`, whose rank is 2 for every non-zero `v_w`.
Rewriting the data can only change *which* plane is constrained, never how many dimensions.
Dropping to rank 1 requires contracting the residual and the Jacobian with a fixed direction —
a change to the measurement *function*, not to its input. The two modifications are therefore not
alternatives; they are orthogonal.

## 8. Composing them

Note that the projection axis is **unchanged** by preprocessing, and not merely parallel:

```
a_hat x m_perp,w = a_hat x ( m_w - a_hat (a_hat . m_w) ) = a_hat x m_w
```

So "preprocess, then project" is literally the same projection applied to the preprocessed
residual. Its retained scalar is

```
dh_s = -|m_perp| ( delta . ( p_hat x w_hat ) ) = -|m_perp| ( delta . a_hat )
```

using `p_hat x w_hat = a_hat`. One line proves it in general: the constrained rotation direction
is proportional to `m_perp,w x w_hat`, and both factors are perpendicular to `a_hat`, so their
cross product is parallel to `a_hat`. **Pure heading, rank 1** — the only such object available,
and the relative-filter form of the classic tilt-compensated compass.

Contrast with the plain projection, where the constrained direction is `m_w x w_hat`: `m_w` has an
`a_hat` component, which is exactly the `cos(theta)` contamination of §6.

## 9. Summary of the geometry

| arm | rank | constrains | pure-tilt vote (ii) | heading purity |
|---|---|---|---|---|
| `mag_on` | 2 | `delta.t_hat`, `delta.w_hat` | yes, full weight | mixed |
| preprocess the field | 2 | `delta.a_hat`, `delta.w_hat` | yes, half weight | pure |
| project the residual | 1 | `delta.t_hat` | **no** | mixed |
| both | 1 | `delta.a_hat` | **no** | **pure** |

Measured residual response per unit relative rotation about each axis, confirming every row
(`|m_w| = 0.529`, `theta = 148.4 deg`):

| rotation axis | `mag_on` | preprocessed | projected | both |
|---|---|---|---|---|
| `a_hat` (heading) | 0.2773 | 0.2773 | 0.2773 | 0.2773 |
| `w_hat` (tilt) | 0.5286 | 0.2773 | **0** | **0** |
| `p_hat` (tilt) | 0.4500 | **0** | 0.4500 | **0** |
| `m_hat` (field) | 0 | 0.2361 | 0 | 0.2361 |

All four preserve the heading response exactly, because `|a_hat x m_perp| = |m_perp| = |m_w|
sin(theta) = |a_hat x m_w|`. The `m_hat` entries are not a leak: a rank-1 constraint on
`delta . a_hat` responds to any `delta` with an `a_hat` component, and `m_hat . a_hat = cos(theta)`.

## 10. What actually happens in the filter

Six arms, geodesic RMSE in degrees against each dataset's ground truth, filter seeded from mocap
at `t = 0` (`run_window`), first 10 s discarded. `proj_pure` is §8's shared-reference version done
inside the filter; `pre` and `pre+proj` are the own-`a_hat` preprocessing of §7, driven through
`mag_override_*`.

**Al Borno — 11 subjects, 19 trials, 131 joint-trials.**

| joint | off | on | proj | proj_pure | pre | pre+proj | best |
|---|---|---|---|---|---|---|---|
| L_Ankle | **8.40** | 14.17 | 11.60 | 12.01 | 12.03 | 11.89 | off |
| R_Ankle | 9.12 | 10.29 | **8.15** | 8.18 | 8.27 | 8.19 | proj |
| L_Hip | 12.85 | 6.55 | 6.28 | 6.28 | 5.97 | **5.92** | pre+proj |
| R_Hip | 27.64 | 11.35 | 11.21 | 9.83 | **9.58** | 9.61 | pre |
| L_Knee | 10.37 | 7.38 | 7.18 | 7.07 | **6.84** | 6.88 | pre |
| R_Knee | 10.43 | 9.39 | 9.07 | 9.04 | 8.62 | **8.60** | pre+proj |
| Lumbar | 35.40 | 7.01 | 7.04 | **5.96** | 6.05 | 6.07 | proj_pure |

Every restricted arm beats `mag_on`; the mixed projection wins at the ankles, the purified arms at
the proximal joints. `proj_pure` reproduces `pre` to within noise (51/131 cases better, mean
+0.14 deg, median +0.07), which is the point: **the active ingredient is the purity of the
attribution direction, and it does not matter whether you reach it by rewriting the reading or by
re-attributing the row.** The shared reference does NOT beat the own-`a_hat` version as §8
predicted it would -- it is ahead only at the two ankles (-0.02, -0.10 deg), where the
manufactured-residual defect should bite hardest. Directionally right, too small to matter.

**IMoVE -- 26 subjects, 236 trials, 1290 joint-trials.**

| joint | off | on | proj | proj_pure | pre | pre+proj | best |
|---|---|---|---|---|---|---|---|
| L_Ankle | 13.83 | **11.24** | 11.33 | 11.85 | 11.75 | 11.92 | on |
| R_Ankle | **12.06** | 15.12 | 14.69 | 15.27 | 15.19 | 15.19 | off |
| L_Hip | 20.37 | 9.95 | 10.13 | **8.53** | 10.16 | 10.02 | proj_pure |
| R_Hip | 14.92 | 9.06 | 9.41 | 9.05 | 8.79 | **8.70** | pre+proj |
| L_Knee | 7.71 | 7.54 | 7.19 | **7.00** | 7.16 | 7.25 | proj_pure |
| R_Knee | 8.05 | 6.23 | 6.22 | 6.22 | 6.12 | **6.17** | pre |

**It does not replicate.** Mean deltas against `mag_on` run -0.44 to +0.67 deg with per-case win
rates near half (95-170 of 194-236), i.e. coin flips. The one real effect is `proj_pure` at the
left hip: -1.43 deg, 173/194 cases, and 184/194 against `proj`.

## 11. Why it does not replicate, and the decision rule that falls out

Run §5's filter-free diagnostic on both datasets and the null is *predicted* rather than
surprising. What the whole argument rests on is the magnetometer's sustained tilt error being
large **relative to the accelerometer's**, because the accelerometer is what the restriction
defers to:

| | acc slow tilt demand | mag slow tilt demand | ratio | mean gain of `proj` vs `mag_on` |
|---|---|---|---|---|
| Al Borno | 1.44 deg | 6.38 deg | **4.26** | **-0.80 deg** |
| IMoVE | 3.45 deg | 4.17 deg | **1.17** | -0.03 deg |

The premise is simply false on IMoVE, and not because its magnetometer is better -- 4.17 against
6.38 deg is a modest difference. It is because **its accelerometer is 2.4x worse** in the same
currency. There is nothing to gain by deferring to a sensor that is no better than the one you
are silencing.

Per joint, pooled over both datasets (n = 13):

```
gain of proj vs mag_on   vs  mag/acc slow-tilt ratio     r = -0.94
                         vs  mag slow tilt alone         r = -0.92
                         vs  acc slow tilt alone         r = +0.35
```

So the deliverable is not the arm, it is the **precondition**: the restriction is worth applying
where the magnetometer's 1 s-block tilt demand exceeds the accelerometer's by roughly 3x or more,
and is not worth applying otherwise. That is measurable on any new dataset in one pass, with no
filter run and no ground-truth orientation beyond what the joint-centre projection already needs
-- `scratch/mag_heading_only.py --dataset <name>`.

Two honest limits on that rule. It is fitted on 13 points from two datasets, one of which supplies
the entire high-ratio end, so the 3x threshold is a reading of the scatter and not a calibrated
boundary. And IMoVE's larger accelerometer demand is itself partly a ground-truth artifact -- its
IMU and mocap are not time-aligned to better than order 1 s per trial before sync, and the
joint-centre projection is driven by marker-derived offsets -- so some of what is scored as
accelerometer disagreement there is reference error, which would also blunt any real improvement.
Those two candidates are not separated.

## 12. Where the code is

| | |
|---|---|
| projection axis | `src/RelativeFilterPlus.py::_heading_projection_axis` |
| shared-reference stripping | `src/RelativeFilterPlus.py::_heading_only_readings` |
| applied to residual | `RelativeFilterPlus.get_h`, the `heading_only_sensor` branch |
| applied to Jacobian | `RelativeFilterPlus.get_H_jacobian`, same branch |
| compiled mirror | `src/relative_filter_fast.py`, the `si == heading_only` block |
| pipeline entry | `experiment_utils._run_relative_filter(heading_only_mag=, heading_only_pure_mag=)` |
| degeneracy gate | `HEADING_AXIS_MIN_SINE = 0.1` in both modules |
| filter arms and scoring | `scratch/mag_heading_only_pilot.py`, report in `scratch/mag_heading_only_report.py` |
| mechanism diagnostic | `scratch/mag_heading_only.py` |
| tests | `test/TestRelativeFilter.py`, `test/TestRelativeFilterFast.py` |

Reference and kernel agree to 7e-14 deg over a trial on every path, including zeroed and
per-sample-gated magnetometer and both attribution modes.

**Three deliberate approximations**, each pinned by a test rather than left to be discovered:

- **The axis is frozen.** `w_hat` comes from the current estimate but is treated as data, so `H`
  is `dh/d_eta` with `w_hat` held fixed. Differentiating exactly would add a `dw_hat/d_eta` term,
  first order rather than second because the residual is non-zero. It is what a tilt-compensated
  compass does, the axis being where the accelerometer says its own null direction lies.
- **The stripped reading is frozen too, under `heading_only_pure`.** Its `H` is the derivative of
  "treat the stripped reading as body-fixed data", NOT of the raw-field measurement -- whose
  derivative is the mixed `t_hat` row. The two arms therefore read one identical number and model
  it differently; neither `H` is more correct a priori, which is exactly why §10 had to be
  measured.
- **The row is dropped, not normalised, near degeneracy.** The sine IS the retained sensitivity,
  and the axis direction degrades as `1/sin`: ~1% magnetometer noise puts its angular uncertainty
  near 6 deg at the gate. A zeroed reading (`mag_off`, `mag_adapt`) takes the same path, which is
  what keeps those a no-op rather than a NaN.

## 13. Open

- Nothing here is a `METHODS` base yet; adding one makes `benchmark_experiment` run it by default
  on every dataset.
- The ratio rule wants a third dataset to be worth trusting. `imove_biplane` cannot supply one --
  its MC10 BioStamps have no magnetometer at all.
- Both levers are orthogonal to `mag_adapt`, which gates *when* the magnetometer speaks rather
  than *what about*. The cross has not been run.
- At the left ankle on Al Borno nothing with a magnetometer beats `mag_off` (8.40 against 11.60).
  Where the magnetometer's own HEADING demand is biased -- 8-10 deg there -- no restriction of
  this kind helps, because that is the vote all of them deliberately keep. Distal segments want
  gating or a joint constraint instead.
- Al Borno's `calcn_l_imu` on `walking` carries two unexplained marker discontinuities (frames
  59826, 59857). R_Ankle shows the same effects independently, so no conclusion rests on it, but
  the size of the L_Ankle numbers partly does.
