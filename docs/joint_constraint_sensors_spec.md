# Joint models as sensors in a relative filter — experiment spec

**Question.** Should a joint model be fed to an orientation filter as a measurement? If so, at
which joints, and which model?

**Decision this produces.** A per-joint recommendation of the form *"at the ankle, use a hinge
constraint at σ = X rad from a cohort axis; at the hip, use nothing"* — plus the evidence for
why, which has to be a mechanism and not a table of RMSEs.

---

## 0. The claim structure

Three quantities decide whether a constraint helps at a given joint, and the study is organised
so each is measured separately rather than inferred from the total.

| | what it is | where it is measured | already done? |
|---|---|---|---|
| **Fidelity** | how wrong the model is — the error floor no filter can beat | `experiments/joint_dof.py`, cross-validated residual per model per joint | **yes** |
| **Information** | how much of the *unobservable* direction the residual spans | new: projection of the constraint Jacobian onto the relative-heading direction | no |
| **Realisability** | how much worse the constraint gets when its parameters come from IMU data instead of mocap | new: the axis-provenance ladder (§4) | partly (`find_biaxial_joint_axes`) |

The predicted benefit is roughly **information × (1 − fidelity cost)**, and the study's job is to
show that this product predicts the measured benefit. If it does, "at which joints" has a
principled answer that transfers to joints and datasets not tested. If it doesn't, we have an
RMSE table and nothing else.

**Why the unobservable direction is the whole story.** With the magnetometer off, the relative
filter sees two accelerometers, which fix gravity in each segment frame — two of the three
relative DOF. Relative heading (rotation about gravity) is unobservable and drifts on the gyros.
Every constraint below is judged by how much of *that* it supplies. This is already written down
in `scratch/knee_axis_sensor.py:1-8`; the study generalises it past one joint.

**Why this matters most on `imove_biplane`.** That dataset's MC10 BioStamps have **no
magnetometer** (`TrackingSpec.has_magnetometer=False`), and its reference is biplane fluoroscopy
on bone. It is simultaneously the case where a constraint is most needed and the only place we
can score against bone rather than skin.

---

## 1. Inventory: what exists, what must be built

### Exists and is reusable

- **Hinge and universal constraints in the filter.** `src/RelativeFilterPlus.py:110-141`
  (`joint_type='1dof'` / `'2dof'`), with `get_h`, `get_H_jacobian`, `get_M_jacobian` blocks.
- **All four joint models fitted, cross-validated.** `experiments/joint_dof.py` — `fit_weld`,
  `fit_hinge`, `fit_universal`, `fit_knee_coupling`, with Reuben's coefficients at
  `joint_dof.py:367`, per-session `pooled_fits.parquet` and cohort `generic_axes.parquet`.
- **Reuben curve evaluation and its derivative.** `_reuben_design` (`joint_dof.py:954`) returns
  `(c(q), dc/dq)` exactly, with the flexion clip. Directly reusable as the coupling measurement.
- **SO(3) plumbing.** `_log_matrix`, `_exp_matrix`, `_right_jacobian`, `_tangent_basis` in
  `joint_dof.py:589-667`.
- **IMU-only axis identification.** `PlateTrial.find_biaxial_joint_axes` (`PlateTrial.py:375`) —
  Gauss-Newton on `(ω_c − ω_p) · (j1 × j2) = 0`. This is the realisable axis source.
- **Method-name suffix grammar.** `resolve_method_spec` (`experiment_utils.py:375`) already
  composes base × normalisation × threshold × distortion × oracles. Constraints slot in as
  another suffix group; no new dispatch mechanism is needed.
- **A pilot with results.** `scratch/knee_axis_sensor.py` — knee only, Al Borno only, standalone.
  Its findings are inputs to this spec (§3.3, §5).

### Must be built

| # | item | est. size |
|---|---|---|
| B1 | Coupling constraint (`'coupling'`) in `RelativeFilterPlus` | ~80 lines + tests |
| B2 | Neutral-angle spring constraint (`'spring'`) in `RelativeFilterPlus` | ~40 lines + tests |
| B3 | Allow **multiple simultaneous** constraints (currently mutually exclusive) | ~30 lines |
| B4 | Constraint support in the numba kernel `src/relative_filter_fast.py` | ~250 lines (see §7) |
| B5 | Constraint parameter resolution: joint → (model, axes, σ) from a provenance | ~150 lines |
| B6 | `experiments/joint_constraint_sensors.py` — the experiment | ~800 lines |
| B7 | Anatomical FE/AA/IE decomposition for reporting (§6) | ~100 lines |
| B8 | IMoveLab global baseline arm | see §5 |

---

## 2. Bugs in the existing 2-DOF constraint — **both fixed, 2026-08-16**

Two independent defects, in the same constraint, neither reachable from `experiments/` because
nothing there has ever set `joint_type`. Both are fixed and both now have tests with teeth.

### 2a. `sin` where the geometry wants `cos`



`RelativeFilterPlus.__init__` stores `sin_alpha = np.sin(dof2_angle_rad)` and `get_h` compares
`v·u − sin_alpha` (`RelativeFilterPlus.py:134`, `:374`). The standard 2-DOF constraint is
`v·u = cos α` for α the angle between the axes. Under the `sin` reading, α must mean *deviation
from orthogonality*, in which case the default `dof2_angle_rad = np.pi/2` enforces `v·u = 1` —
coincident axes, which is not a universal joint under either convention.

Nothing in `experiments/` currently uses `joint_type`, so this has never fired.
`scratch/knee_axis_sensor.py:356` works around it by passing
`arcsin(target_dot)`. `test/TestRelativeFilter.py:278` asserts `sin(alpha)`, i.e. it tests the
implementation against itself.

**Done:** switched to `cos α`; `dof2_angle_rad` is now required (no default, so the degenerate
π/2 cannot be reached by omission); `test_2dof_residual_vanishes_at_the_true_carrying_angle`
asserts the geometry over α ∈ {0, π/6, π/3, π/2, 2π/3, π} rather than re-deriving the
implementation's expression; the pilot's `arcsin` workaround became `arccos`, which reproduces
the identical constraint, so the pilot's numbers stand.

### 2b. The 2-DOF Jacobian was negated and in the wrong frame

Found by numerically differentiating `get_h` while verifying the two new constraints — the
existing analytic `H` was off by a **relative error of ~1.0**, i.e. entirely wrong. It computed
the bare world-frame cross products `u_w × v_w` and `v_w × u_w`. The error state is a
*body-frame* right-multiplied perturbation, so the correct blocks are

```
∂h/∂η_p =  R_wpᵀ (v_w × u_w)
∂h/∂η_c = −R_wcᵀ (v_w × u_w)
```

— the old version was missing both `Rᵀ` factors *and* negated. A sign-flipped `H` drives the
correction the wrong way, so this is not a small-accuracy issue.

It survived because `test_2dof_joint_residual_and_shapes` asserted only `H.shape == (7, 6)`,
while the file's existing `numerical_H` helper was applied to the 1-DOF joint and never to the
2-DOF one. `test_every_joint_constraint_jacobian_matches_numerical` now covers all four
constraints over six random poses at `atol=1e-5` (achieved: ~1e-9).

**Consequence for the pilot.** `scratch/knee_axis_sensor.py`'s universal-joint results were
produced with this Jacobian. Its headline 2-DOF finding — 104° at σ = 0.0003, read as "a
near-hard scalar constraint on a slightly wrong carrying angle drags the estimate with it" —
is now **suspect**: a negated gradient is at least as good an explanation for a blow-up. The
1-DOF numbers are unaffected (that Jacobian was correct and tested). **Re-run the pilot's 2-DOF
arm before quoting any universal-joint number**, including the "universal is nearly exact but
barely constrains anything" framing in §9's predictions.

---

## 3. The four constraints, as EKF measurements

State is the 6-D error rotation `η = [η_p, η_c]`; `R_pc = R_wp^T R_wc`. All residuals are
evaluated at `η = 0` and stacked below the accelerometer/magnetometer blocks in `h`, `H`, `R`.

### 3.1 Hinge (1-DOF) — exists

```
r = R_wp y_J − R_wc y_K                       ∈ ℝ³, rank 2
H = [ R_wp [y_J]ˣᵀ , −R_wc [y_K]ˣᵀ ]          3×6
```

Constrains **two** of three relative DOF. The residual is a difference of unit vectors, so
‖r‖ ≈ misalignment in radians. **σ is dimensionless radians and must not be set from
`DEFAULT_ACC_STD`**, which lives on the m/s² scale.

### 3.2 Universal (2-DOF) — exists, after §2

```
r = (R_wp v_J)·(R_wc u_K) − cos α             ∈ ℝ¹
H = [ (u_K^w × v_J^w)ᵀ , (v_J^w × u_K^w)ᵀ ]   1×6
```

Constrains **one** DOF. `d(cos)/dθ = −sin θ ≈ 1` at near-orthogonal carrying angles, so the same
radian scaling applies to first order.

### 3.3 Knee coupling (1-DOF nonlinear) — **new**

The published knee: one angle per sample, the two off-axis channels pinned to Reuben's quartics
instead of to zero. Using `joint_dof`'s gauge (the component of the rotation vector along `u_c`
*is* the joint angle):

```
ν  = log( R_0ᵀ R_pc )                          ∈ ℝ³
q  = ν · u_c                                   (the flexion angle, by gauge)
P  = tangent_basis(u_c)                        3×2
r  = Pᵀ ν − c(q)                               ∈ ℝ²        ← the two coupled channels
```

Jacobian. With `M = R_0ᵀ R_pc = exp([ν])`, perturbation sends
`M → exp([−R_0ᵀ η_p]) M exp([η_c])`, so

```
δν     = −J_l(ν)⁻¹ R_0ᵀ η_p + J_r(ν)⁻¹ η_c ,      J_l(ν) = J_r(ν)ᵀ
∂r/∂ν  = Pᵀ − c′(q) u_cᵀ                          2×3
H      = (Pᵀ − c′(q) u_cᵀ) [ −J_r(ν)⁻ᵀ R_0ᵀ , J_r(ν)⁻¹ ]      2×6
```

`c` and `c′` come from `_reuben_design(q, σ_flex)` unchanged, including the flexion clip at zero
(outside which the quartics diverge rather than extrapolate). Parameters: `u_c`, `R_0`, the
channel-frame map and flexion sign — all persisted by `fit_knee_coupling` (`joint_dof.py:1840`).

**Note this is a strictly stronger constraint than the hinge** (2 residual dims pinning
*specific* values, vs 3 dims of rank 2 pinning axis alignment). It is not "hinge plus a bit".

### 3.4 Neutral-angle spring — **new**

```
ν = log( R_0ᵀ R_pc )
r = ê_IE · ν                                   ∈ ℝ¹, driven to 0
H = ê_IEᵀ [ −J_r(ν)⁻ᵀ R_0ᵀ , J_r(ν)⁻¹ ]        1×6
```

**Gain-to-σ mapping, so we can run at the paper's strength.** The paper applies a feedback
correction of gain κ. A scalar Kalman update with prior variance p and measurement variance σ²
applies a fraction `p/(p + σ²)` of the residual, so matching gives

```
σ² = p (1 − κ) / κ
```

κ = 0.0064 (hip) → σ² ≈ 155 p — barely on. κ = 0.25 (ankle) → σ² = 3 p. Report both the
paper-matched σ and the swept σ; if the swept optimum is far from the paper-matched value, that
is a finding about their tuning, not a mismatch to hide.

**`ê_IE` needs a definition and this is an open call.** Plate frames are not anatomical on the
marker datasets (`joint_dof.py:120-127`). Default proposal: the child segment's long axis, taken
as the universal fit's `u_K`, which is child-fixed by construction and is the internal/external
rotation axis at both hip and ankle. Alternative: the joint-centre-to-distal direction from
`experiments/joint_center.py`. Run both in the pilot phase and pick one.

---

## 4. The axis-provenance ladder

**This is the credibility axis of the whole study.** Every constrained arm runs at every rung.

| rung | source | what it assumes | role |
|---|---|---|---|
| `oracle` | this subject's own axes fitted to this trial's mocap (`joint_dof` per-trial fit) | ground truth at run time | **ceiling only** — never quoted as a result |
| `session` | this subject's axes pooled over the session (`pooled_fits.parquet`) | one mocap calibration per session | realistic if a calibration trial is allowed |
| `cohort` | leave-one-subject-out axial mean (`generic_axes.parquet`) | a published generic axis | the "no calibration" case |
| `inertial` | `find_biaxial_joint_axes` on IMU data alone, this trial | a calibration *movement*, no mocap | **the deployable case, and the headline number** |
| `unit` | nearest coordinate direction of each plate frame | nothing | crudest floor |

Two things this ladder buys that a single provenance cannot:

1. **`oracle − inertial` is the calibration cost**, per joint. Given
   `imove-imu-offset-excitation-bias` (offset fits are trial-dependent rather than
   excitation-driven), expect this gap to be large and unevenly distributed. If it swallows the
   whole benefit at a joint, that joint's answer is "no" regardless of fidelity.
2. **`cohort` vs `session`** answers "does this need a per-subject calibration or can we ship a
   number?" — which is the question a practitioner actually has.

`generic_axes.parquet` already reports `axis_from_cohort_deg` (`joint_dof.py:2227`), so the axis
dispersion that drives the `cohort` rung's penalty is already quantified.

---

## 5. Arms

### 5.1 Relative-filter arms

Method names extend the existing suffix grammar. Proposed form:

```
mag_off_hinge_cohort_s0.010
mag_off_coupling_inertial_s0.006
mag_on_universal_session_s0.020
```

i.e. `<base>_<model>_<provenance>_s<constraint_std>`, appended after the existing normalisation
suffix and before the oracle suffixes. `resolve_method_spec` gains one group in
`_METHOD_SUFFIX_RE` and three keys in the spec dict.

**Constraint models:** `none`, `hinge`, `universal`, `coupling` (knee only), `spring` (hip and
ankle only), and `hinge+spring` / `universal+spring` once B3 lands.

**Bases:** `mag_off` is the primary — it is the case where heading is unobservable and therefore
where the question is live. `mag_on` runs as a control: if a constraint helps there too, the
mechanism is not "replacing the magnetometer" and the story changes. On `imove_biplane`,
`mag_off` is the only available base.

### 5.2 The constraint σ sweep — mandatory, not optional

The pilot establishes that **σ cannot be set from the measured model error**. On Subject09's
L_Knee, the model-error σ (0.10–0.14 rad) gives 4.30° against a 4.35° `mag_off` baseline — the
constraint is drowned — while σ = 0.01 rad gives 3.19°. At σ = 0.0003 the 2-DOF arm blows up to
104°. The reason is that model error is not white: it is soft-tissue motion and secondary
rotation, correlated over many samples, so its per-sample scatter overstates the uncertainty in
its mean direction.

**Grid:** `[0.003, 0.006, 0.01, 0.02, 0.04, 0.08, 0.16]` rad, as the pilot. Report the whole
curve. Quote a **single common σ chosen once** on a held-out slice and applied to every arm,
joint and dataset — picking each arm's best σ on the data it is scored on is exactly the
optimism that killed the `o^J` gating result (`o-j-gating-does-not-beat-mag-on`).

**Watch for endpoint optima.** If every joint's optimum is at a grid endpoint, that is the same
pathology as the o^J threshold study and means the constraint is acting as a binary on/off rather
than as a weighted sensor. Check this explicitly and report it.

### 5.3 The global baseline (IMoveLab)

Yes, run it — and note that `data/IMoveLab_Raw_Data` means we are on **their own data**, so this
is apples-to-apples in a way that is unusual and worth exploiting.

Three arms, in increasing order of what they claim:

1. **`ekf`** — the existing absolute baseline, unconstrained. Already implemented
   (`_setup_ekf_ground_plate_`).
2. **`ekf_imovelab`** — their constraints in their form: a **post-update feedback correction** at
   gains α = 0.9 (knee coupling), κ = 0.0064 (hip), κ = 0.25 (ankle). *Not* a measurement update.
   At 100 Hz, α = 0.9 is replacement rather than regularisation — `joint_dof.py:54` already says
   so, and it means their knee arm essentially *defines* two of three knee DOF from flexion.
3. **`ekf_imovelab_measurement`** — the same constraints recast as EKF measurements at the
   matched σ from §3.4. The difference between arms 2 and 3 isolates *feedback vs measurement*,
   which is a confound we otherwise carry into every comparison.

**Reproduce before you compare.** Their published biplane numbers are knee FE 2.8–2.9° RMSD
(from 4.1–4.2° unconstrained), AA 0.9°, IE 3.7°, and <5° over a 70-minute walk against ~40° for
unconstrained filters. If our reimplementation of arm 2 does not land near those on the biplane
trials, the reimplementation is wrong and no comparison against it means anything. Their code is
at `github.com/CMU-MBL/IMoveLab`; running it directly on our built trials is the stronger option
and I'd take it if the interface allows.

### 5.4 Grid size

```
2 bases × 6 constraint models × 5 provenances × 7 σ  = 420 relative arms (full crossing)
```

That is far too many. Reduced design:

- **σ sweep** at one provenance (`session`) and one base (`mag_off`): 6 × 7 = 42 arms.
- **Provenance ladder** at the common σ: 6 × 5 = 30 arms.
- **`mag_on` control** at the common σ, `session` only: 6 arms.
- **Global baseline**: 3 arms.

**81 arms**, of which the σ sweep is by far the biggest and is the one that most needs the
compiled kernel.

---

## 6. Datasets, and what each is for

| dataset | reference | mag? | duration | role in the study |
|---|---|---|---|---|
| `alborno` | marker plates | yes | walking + complexTasks | **the main grid.** 7 joints × 11 subjects; the only place all joints are present |
| `imove_biplane` | biplane fluoroscopy, bone | **no** | 0.4–0.6 s hops | **mechanism, against bone.** Only place model error is separable from soft-tissue artifact |
| `imove_biplane_vicon` | Vicon cluster, same trials | no | same | the marker-vs-bone delta; already the reference-agreement path in `joint_dof` |
| `imove` | marker plates | yes | incl. long walks (100 Hz sensors) | **drift.** The 70-minute case where a constraint's value is a drift bound, not an RMSE |

**Read the biplane result carefully.** 0.4–0.6 s of one hop is too short for heading drift to
accumulate, so the biplane trials will *understate* the constraint's benefit while giving the
only trustworthy read on its fidelity. The long walks are the reverse. Neither alone answers the
question; the pair does.

---

## 7. Performance, and the phasing it forces

`src/relative_filter_fast.py` hardcodes `NUM_VECTOR_SENSORS = 2` and has no joint block, so every
constrained arm currently falls back to the Python reference at ~1/55 the speed — roughly 20 s
per joint-trial at 60 k samples. Al Borno alone is ~154 joint-trials, so **~50 minutes per arm**,
and 81 arms is ~68 hours. That is not viable for the sweep.

### 7.1 All four constraints fit in one kernel — and that reorders the phases

The answer is **yes, all of them, in a single kernel with a single specialization**. Three things
make it cheaper than the estimate above:

**No matrix logarithm is needed.** The kernel already carries the state as scalar-last
quaternions. `ν = rotvec(q_0⁻¹ ⊗ q_wp⁻¹ ⊗ q_wc)` is a quaternion product plus a rotvec
extraction, and both already exist in the kernel for the state update. The scariest-looking part
of the coupling and spring constraints disappears.

**Pad, don't size.** The real blocker is that constraints change the measurement dimension
(6 → 7/8/9), and runtime-sizing `S` per step reintroduces exactly the per-step allocation the
kernel exists to avoid. Instead run at a fixed maximum dimension always, and for unused rows set
the `H` row to zero, the residual to zero, and the `R` diagonal to one. That is **exactly**, not
approximately, the smaller system: a zero `H` row makes that column of `P Hᵀ` zero, hence that
column of `K` zero, so the row contributes nothing to `K e` or `K H`, and the unit diagonal keeps
`S` invertible. One kernel, no branching in the hot loop, no allocation. The cost is a 10×10
solve instead of 6×6 — a few hundred flops against the ~30 interpreter dispatches it replaces.

**Multi-constraint (B3) comes free.** With padding, `hinge + spring` is just four non-zero
constraint rows instead of three. Max constraint rows = 4, so max measurement dim = 10.

What remains is mechanical: `J_r(ν)⁻¹` in closed form (~20 lines — already written and verified
in `src/toolchest/so3.py`, so the kernel transcribes rather than derives), the Reuben quartic and
its derivative (~15), the hinge and universal blocks (~30), generalizing the existing solve's
loop bounds (~40), and a Python-side parameter packer (~60). The one genuinely ugly part is
numba's dislike of heterogeneous structs: the per-constraint parameters get packed into a flat
`float64` array alongside an integer `constraint_code`, which is standard for njit kernels but
needs its own test.

**Consequence: kernel-first.** Phase A's original job was a go/no-go *before* sinking effort into
the kernel. If the kernel is about a day and removes ~68 hours of compute, that rationale is
gone. The revised order is below. Note that B1/B2 are **not** skipped by going kernel-first —
`test/TestRelativeFilterFast.py` holds the kernel to the reference implementation, so the
reference must have the constraints first regardless. That is why they were built first.

### 7.2 Revised phases

**Phase 0 — reference constraints. DONE (2026-08-16).** `src/toolchest/so3.py` and
`src/joint_constraints.py` extracted from `joint_dof` so `src/` owns the shared formulas without
importing `experiments/`; `joint_dof` re-exports them, and its 64 tests still pass. Coupling and
spring implemented in `RelativeFilterPlus`; both 2-DOF bugs fixed (§2). All four constraint
Jacobians verified against numerical differentiation at ~1e-9, now permanently tested.

**Phase A′ — settle the open modelling calls. DONE (2026-08-16).** Results in §7.3.

**Phase B — kernel**, per §7.1.

**Phase C — full grid.** All datasets, all 81 arms.

**Phase D — baseline.** IMoveLab reproduction and the three global arms.

### 7.3 Phase A′ results

**The 2-DOF blow-up was the bug (§2b), not geometry.** Re-running the pilot on Subject09 with
the corrected Jacobian, measured as improvement over that run's own `mag_off` (which cancels an
unrelated pipeline shift — see the caveat below):

| arm | σ = 0.003, old J | σ = 0.003, fixed J |
|---|---|---|
| `2dof_cohort` | **−1.95°** (worse than baseline) | **+0.77°** |
| `2dof_unit` | **−4.92°** | **+0.72°** |
| `2dof_per-subject` | +0.44° | +0.64° |

Every 2-DOF arm now improves monotonically as σ tightens and none falls below baseline. The
pilot's published explanation — "a near-hard scalar constraint on a slightly wrong carrying
angle drags the whole estimate with it" — is **retracted**; it described a negated gradient.
The largest shift was 22.8° on `2dof_cohort`, against a ≤3.0° confound floor on the 1-DOF arms.

**What survives:** 2-DOF buys little even when correct. Best gain over `mag_off` is +0.7 to
+1.3°, against +2.0 to +3.9° for the 1-DOF arms — so §9's "universal < hinge everywhere"
prediction stands, now for the stated reason (one scalar residual against two constrained DOF)
rather than by accident.

**Caveat on absolute numbers.** `mag_on` and `mag_off` *also* moved by ~+5° between the archived
pilot artifact and today, and neither carries a constraint — so some other pipeline change
landed in between and the old parquet is not a valid control in absolute terms. Only
improvement-over-baseline is comparable across the two runs. The pilot's docstring numbers
(4.35° baseline, 3.19° at σ=0.01) no longer reproduce; the current baseline is 8.43°. Tracking
down that ~5° shift is worth doing before Phase C, since it moves every number in the study.

**`ê_IE` = the segment long axis, at both joints.** `scratch/spring_axis_choice.py` compares the
two candidates on merit = heading gain / residual RMS (information per degree of model error),
over 11 subjects and 18–19 trials per joint, with no filter in the path:

| joint | segment | universal | paired result |
|---|---|---|---|
| R_Hip | **0.154** | 0.041 | segment wins **18/18** |
| L_Hip | **0.133** | 0.037 | segment wins **19/19** |
| R_Ankle | 0.040 | **0.050** | 7/19 — inside 1 SD |
| L_Ankle | 0.051 | **0.053** | 9/18 — inside 1 SD |

At the hip `segment` dominates on *both* components (RMS 6.6–7.5° vs 14.6–14.7°; gain 0.88 vs
0.32) and wins every single trial. At the ankle the two are statistically indistinguishable.
**Decision: `segment` at both**, since it is unanimous where there is a difference and a
coin-flip where there isn't — one definition, no per-joint knob tuned on scored data.

Two findings that fell out and matter beyond the axis choice:

1. **The spring is a hip term, not an ankle term** — on this data. Heading gain is 0.88 at the
   hip against 0.19–0.33 at the ankle, and merit is ~3× higher. IMoveLab weights it the other
   way (κ = 0.25 ankle, 0.0064 hip — nearly off at the hip). This is a tension, not a
   contradiction: their gains were tuned for an *absolute* filter where heading is observable
   from the magnetometer, so "how much of the unobservable direction does this see" is not the
   quantity they were optimising. It does mean their κ values should not be carried over to the
   relative filter unexamined, and §5.3's paper-matched arm should be read as *their* tuning
   rather than *a* tuning.
2. **The neutral is a second open call, hiding inside the axis question.** With `R_0` the
   trial-mean relative rotation, `mean(ν · ê) ≡ 0` for every axis by construction — so the
   spring pulls toward *this trial's own mean pose*, not toward anatomical neutral, which is
   what IMoveLab's term does. The trial-mean version also cannot be computed causally. Neither
   is available on a plate frame without a calibration pose. Added to §12.

---

## 8. Metrics

**Report per-DOF, not pooled 3-D angle.** `compute_error_stats` already emits `X/Y/Z` components
of the error rotation vector plus `MAG` — but those are *plate-frame* components, not FE/AA/IE.
The entire hypothesis is that constraints act on specific anatomical DOF (heading ≈ IE at the
hip and ankle), so a pooled scalar hides the mechanism and an unrotated component triple is
uninterpretable across subjects. **B7: add an anatomical decomposition** using each joint's fitted
`R_0` and primary axis from `joint_dof`, reporting error in (FE, AA, IE). This also makes the
numbers directly comparable to the paper's 2.8 / 0.9 / 3.7°.

Primary outcomes, per joint × dataset × arm:

1. **RMSE per anatomical DOF**, and pooled `MAG` for continuity with existing figures.
2. **Heading-drift rate** — error about gravity, regressed on time over the long walks. This is
   what a constraint is actually buying and it is invisible in a short-trial RMSE.
3. **Information score** — mean over samples of the projection of the constraint's row space onto
   the relative-heading direction. Computable with no filter at all, from mocap plus the axes.
   This is the "information" column of §0 and it is what makes the recommendation transferable.
4. **Fidelity** — the cross-validated model residual from `joint_dof`, carried across unchanged.

Then the one plot the study exists for: **measured benefit (mag_off RMSE − constrained RMSE)
against predicted benefit (information × fidelity)**, one point per joint per dataset. If that
correlates, we can answer "which joints" for joints we never ran.

---

## 9. Pre-registered predictions

Write these down before Phase C and score against them.

- **Ankle: yes.** Highest heading unobservability (the foot is near-static in stance so the
  accelerometer is informative but the heading is not excited), and the talocrural joint is the
  closest thing to a real hinge in the lower limb.
- **Knee: hinge helps, coupling does not help more.** The pilot already has the knee hinge at
  3.19° vs 4.35° `mag_off`. Coupling adds curvature only — a *linear* coupling is a hinge about a
  tilted axis and the hinge's axis is already free (`joint_dof.py:64-68`), so Reuben's entire
  budget is the curvature of the quartics over the visited flexion range.
- **Hip: no.** Genuinely 3-DOF; the spring at κ = 0.0064 is near-inert by construction.
- **Universal < hinge everywhere**, because 1 scalar residual against 2 constrained DOF, despite
  universal's much better fidelity (1.6–1.8° vs 7.9–9.5° at the knee). The pilot's framing —
  "the more informative constraint is the more wrong model" — is the tension the study resolves.
- **`inertial` provenance loses 30–60% of the `oracle` benefit**, unevenly, worst where excitation
  is lowest.

---

## 10. Confounds and controls

- **Burn-in.** Set `init_orientation_std` explicitly on every arm. Leaving `P = eye(6)` makes the
  first update apply nearly the whole residual and produces a tens-of-seconds transient that
  dominates pooled RMSE — this has already produced one false result in this repo
  (`filter-burn-in-dominates-pooled-rmse`).
- **Output namespace.** `joint_angles/` is shared and clobberable across experiments; write under
  a distinct dataset/method namespace and check the manifests.
- **Circular axes.** `oracle` and `session` provenances fit axes to the same mocap that scores the
  filter. Labelled as ceilings throughout; never quoted as headline numbers.
- **Constraint σ tuned on scored data.** Mitigated by the single common σ (§5.2).
- **World frame.** The biplane half is Z-up while the marker datasets are Y-up
  (`biplane-world-frame-is-z-up`). The relative arms never consult gravity, but every `ekf` arm
  and both acc oracles do, silently.
- **Two Python environments.** Plotting and tests need `venv/bin/python`.
- **Test-suite concurrency.** Do not run the suite alongside a grid; it produces phantom errors.
- **`mag_on` control.** If constraints help as much with the magnetometer on, the mechanism is not
  heading recovery and §0's model is wrong.

---

## 11. Deliverables

- `experiments/joint_constraint_sensors.py` — the grid, per-trial parquets, per-dataset
  statistics, console report, matching the `joint_dof` / `joint_center` house shape.
- `plotting/joint_constraint_sensors.py` — the σ sweep curves, the provenance ladder, and the
  benefit-vs-prediction scatter.
- `test/TestJointConstraintSensors.py` — Jacobians against numerical differentiation, the
  gain-to-σ mapping, the σ unit conversion, and constraint parameter resolution.
- Extensions to `test/TestRelativeFilter.py` and `test/TestRelativeFilterFast.py` for the new
  constraints and the §2 fix.

---

## 12. Open calls

1. ~~**`ê_IE` definition** for the spring.~~ **Settled in Phase A′ (§7.3): the segment long
   axis, at both hip and ankle.**

1b. **What is the spring's neutral?** Raised by Phase A′ and genuinely open. `R_0` = trial mean
   pulls toward this trial's own mean pose and is not causally computable; anatomical neutral is
   what IMoveLab means but is not available on a plate frame without a calibration pose. Options:
   a static-pose calibration, `R_0` from a held-out earlier trial, or a causal running mean.
   This has to be answered before the spring arms can be quoted as deployable.
2. **Run IMoveLab's own code, or reimplement?** Proposal: attempt theirs first; reimplement only
   if the interface fights us, and gate on reproducing their published biplane numbers either way.
3. **Does `mag_on` × constraints stay in scope**, or is the study `mag_off`-only with `mag_on` as a
   6-arm control? Proposal: control only, as costed in §5.4.
4. **Multi-constraint arms** (`hinge+spring`) — in scope for Phase C, or deferred? They require
   B3 and double the model count. Proposal: defer unless Phase A shows both helping at one joint.
