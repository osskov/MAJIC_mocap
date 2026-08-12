# MAJIC_mocap
This repository contains the Python implementation of the "MAJIC Filter," a Kalman filter designed for joint orientation estimation using Inertial Measurement Units (IMUs), as presented in the associated publication. It also includes a complete toolchain for data processing, statistical analysis, and figure generation to reproduce the results from the paper.

## Overview

The core of this project is the `RelativeFilter`, an extended Kalman filter that estimates the relative orientation (joint angles) between two body segments, each equipped with an IMU. The filter is designed to be robust and adaptable to various conditions by selectively incorporating magnetometer data.

This repository provides all the necessary scripts to:
1.  Load and process raw IMU and motion capture data.
2.  Generate segment orientations using different estimation methods.
3.  Calculate joint kinematics and error statistics against a ground-truth reference.
4.  Perform statistical analysis to compare the performance of the different methods.
5.  Generate all the figures presented in the publication.

The data, adapted from Al Borno et al. 2022, can be found on Google Drive (https://drive.google.com/drive/folders/1t5xS7Y1q29BTAcFl0IslfVBKy48U-WMY?usp=sharing) or at SimTK (https://simtk.org/projects/majic_dataset).

## Repository Structure

Inputs and outputs are kept strictly separate. **`data/` is read-only source data** —
nothing in this repository ever writes into it. Every generated artifact goes to
`results/` (tabular data) or `plots/` (figures). All paths are defined in one place,
[`paths.py`](paths.py), and the write helpers there raise if something tries to write
under `data/`, so the rule is enforced rather than merely documented.

```
.
├── data/                                   # <-- INPUTS ONLY, exactly as downloaded
│   ├── Subject01/
│   │   ├── walking/
│   │   │   ├── imu data/                   # Raw IMU .txt files
│   │   │   ├── madgwick (al borno)/        # Outputs from Al Borno et al. (2022)
│   │   │   └── walking.trc                 # Mocap ground truth
│   │   └── complexTasks/                   # ... same structure
│   └── ... (Subject02 … Subject11)
│
├── results/                                # <-- ALL GENERATED DATA
│   ├── joint_angles/
│   │   ├── Subject01/walking/mag_on.parquet
│   │   └── <variant>/Subject01/walking/...      # a run at non-default filter stds
│   ├── statistics/
│   │   ├── all_subject_statistics.parquet      # benchmark summary (paper figures)
│   │   ├── all_subject_joint_angles.parquet    # concatenated time series (~1 GB)
│   │   ├── oracle_ablation_statistics.parquet  # one per named experiment
│   │   └── per_subject/<experiment>/Subject01/walking.parquet
│   └── experiments/
│       ├── noise_sensitivity/
│       ├── drift_observability/
│       └── sensor_distributions/              # per-trial sample tables behind its figures
│
├── plots/                                  # <-- ALL GENERATED FIGURES
│
├── paths.py                                # Every path in the repo, defined once
├── pyproject.toml                          # Package metadata + one console command per script
│
├── src/
│   ├── toolchest/                          # IMUTrace, WorldTrace, PlateTrial, AHRSFilter
│   └── RelativeFilterPlus.py               # Core MAJIC / relative filter
│
├── experiments/                            # One script per experiment (see below)
│   ├── experiment_utils.py                 #   shared physics / IO / orchestration engine
│   └── <name>.py                           #   computes results -> results/statistics/
├── plotting/                               # One script per experiment, same basename
│   ├── utils.py                            #   shared plotting + significance testing
│   └── <name>.py                           #   reads results/statistics/ -> plots/
│
├── test/                                   # Unit tests (see "Tests" below)
├── scratch/                                # Ad-hoc exploration, not part of the pipeline
└── README.md
```

Every artifact is written with a `<name>.manifest.json` sidecar recording the git SHA,
whether the working tree was dirty, the timestamp, the command line, and the physical
constants in force (gravity, noise standard deviations, `mag_adapt` threshold). Outputs
are overwritten in place on re-run, so the sidecar is what tells you which code version
produced a given file:

```json
{
  "git_sha": "0d77bc0…", "git_branch": "paper_revision", "git_dirty": true,
  "written_at": "2026-08-10T17:44:01+00:00",
  "written_by": "benchmark_experiment.py",
  "argv": ["--subjects", "01", "--activities", "walking"],
  "constants": {"expected_gravity": [0.0, 9.81, 0.0], "acc_std": 0.037, "mag_std": 0.03,
                "mag_adapt_threshold": 1000.0},
  "method": "mag_off", "subject": "Subject01", "n_rows": 422275
}
```

-   **`paths.py`**: Single source of truth for every filesystem path. Paths are anchored
    to the repository root, not the working directory, so scripts resolve identically no
    matter where they are invoked from.
-   **`src/RelativeFilterPlus.py`**: The core implementation of the Relative Filter.
-   **`src/toolchest/`**: Utility classes for IMU data (`IMUTrace`), motion capture data
    (`WorldTrace`), and synchronized trial data (`PlateTrial`).
-   **`experiments/experiment_utils.py`**: The shared engine behind every experiment — data loading,
    the oracle/physics helpers, the filter driver, error statistics, and the parallel
    grid runner with its live status table.
-   **`experiments/benchmark_experiment.py`**: The main pipeline. Computes joint angles
    for every method across every subject/activity, then aggregates error statistics
    against the marker (mocap) ground truth into
    `results/statistics/all_subject_statistics.parquet`.

    Note one deliberate asymmetry in `METHODS`: the **EKF normalizes its vector
    measurements and the relative filters do not**. This is a tuning choice, not a units
    choice — normalizing de-weights the accelerometer ~96x relative to the gyro, which the
    EKF needs (56° → 20° pooled RMSE, ankles 94° → 26°) and the relative filters do not
    (`mag_on` 9.5° → 17.1°). The EKF compares one real accelerometer against a virtual
    ground plate reading constant gravity, so foot-strike linear acceleration is pure
    uncancelled error; the relative filter differences two real accelerometers at a shared
    joint center, where it largely cancels. The flag is set on the `ekf` **base**, so
    `ekf_perfect_acc` and the other oracle variants inherit it and the oracle gaps stay
    uncontaminated. See `experiments/normalization_comparison.py` for the three-arm
    evidence.
-   **`plotting/`**: One plotting script per experiment, sharing the experiment's
    basename (`experiments/oracle_ablation.py` -> `plotting/oracle_ablation.py`; the
    benchmark's figures are `plotting/paper_figures.py`). Plotting never recomputes —
    each script reads the statistics file its experiment wrote, so figures can be
    re-tuned without re-running the pipeline. `plotting/utils.py` holds the shared
    distribution/heatmap engine and significance testing.
-   **`plots/`**: The default output directory for all generated figures.
-   **`scratch/`**: Ad-hoc exploration and one-off diagnostic scripts. Not part of the
    reproducible pipeline and not required for any paper figure.

## Installation

To set up the environment and run the scripts, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MAJIC_MOCAP
    cd MAJIC_MOCAP
    ```

2.  **Create a Python environment:** It is highly recommended to use a virtual environment (e.g., venv or conda).

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the project:** An editable install puts `paths`, `src`, `experiments`, and
    `plotting` on the import path, so scripts run from any working directory. Dependencies
    come from `requirements.txt`, which `pyproject.toml` reads directly.
    ```bash
    pip install -e .
    ```

    This also installs a console command for every script below (`majic-benchmark`,
    `majic-plot-paper-figures`, …). They are aliases: `majic-benchmark` and
    `python -m experiments.benchmark_experiment` run the same `main()` with the same
    flags. `pip install -r requirements.txt` still works if you only want the
    dependencies, but then commands must be run from the repository root.

## Data Setup

Download the dataset (see the links above) and unpack it into `data/`, which the scripts
treat as read-only:
```
data/
├── Subject01/
│   ├── walking/
│   │   ├── walking.trc                 # Mocap ground truth
│   │   ├── imu data/                   # Raw IMU .txt files, one per segment
│   │   └── madgwick (al borno)/        # Optional: Al Borno et al. (2022) outputs
│   └── complexTasks/                   # ... same structure
├── Subject02/
│   └── ...
└── ...
```
-   Each subject has their own directory (`Subject01` … `Subject11`).
-   Inside it, one subdirectory per trial type (`walking`, `complexTasks`).
-   Each trial directory must contain a `.trc` motion capture file and an `imu data`
    subdirectory holding the raw per-segment IMU `.txt` files.

Nothing is ever written back into `data/`; `results/` and `plots/` are created on demand.

## Usage and Workflow

To reproduce the results from the publication, run the scripts in the following order.

Every command below is written as `python -m <module>`, which requires the repository
root as the working directory. After `pip install -e .` each one also has an equivalent
console command that works from anywhere — `majic-<name>` for `experiments/`,
`majic-plot-<name>` for `plotting/` (so `python -m plotting.paper_figures` is
`majic-plot-paper-figures`). Run `majic-<tab>` to list them.

### Step 1: Run the benchmark pipeline

Computes joint angles for every method across every subject/activity, then aggregates
error statistics against the marker (mocap) ground truth.

```bash
python -m experiments.benchmark_experiment
```

This writes:

-   `results/joint_angles/Subject<NN>/<activity>/<method>.parquet` — per-method joint
    angles (rotation vectors), one file per subject/activity/method.
-   `results/statistics/per_subject/all_subject/Subject<NN>/<activity>.parquet` —
    per-subject error statistics.
-   `results/statistics/all_subject_joint_angles.parquet` — the concatenated time series
    across all subjects and methods (large, ~1 GB).
-   `results/statistics/all_subject_statistics.parquet` — the summary statistics used by
    the plotting scripts.

Each of these gets a `.manifest.json` provenance sidecar alongside it.

Useful flags: `--subjects`, `--activities`, `--methods` to restrict the grid, `--workers`
to cap parallelism, and `--stats-only` to skip regeneration and re-run just the
aggregation over the joint angles already on disk:

```bash
python -m experiments.benchmark_experiment --subjects 01 02 --activities walking
python -m experiments.benchmark_experiment --stats-only
```

### Step 2: Run any additional experiments (optional)

Each script under `experiments/` is a variation on the same pipeline and writes its own
summary to `results/statistics/<name>_statistics.parquet`:

```bash
python -m experiments.normalized_benchmark     # the benchmark grid, normalized and re-tuned
python -m experiments.oracle_ablation          # acc/mag ground-truth ablation
python -m experiments.threshold_sensitivity    # mag_adapt observability threshold sweep
python -m experiments.distortion_tolerance     # how much magnetic distortion mag_on tolerates
python -m experiments.noise_sensitivity        # gyro/acc/mag noise parameter sweep
python -m experiments.ekf_oracle_comparison    # EKF vs. its oracle variants
python -m experiments.drift_observability      # segment-and-reset drift diagnostic
python -m experiments.acceleration_projection  # validates the joint-center acc projection
python -m experiments.sensor_distributions     # sensor distortion / acceleration / observability
python -m experiments.relative_vs_absolute     # relative vs global-reference correction geometry
python -m experiments.normalization_comparison # unit-length vs raw-magnitude filter inputs
```

`normalized_benchmark` is Step 1 again with two things changed: **every arm normalizes its
vector measurements**, and the noise model is re-tuned to `gyro_std=0.0116`,
`acc_std=0.03`, `mag_std=0.05` (against the shipped `0.0045 / 0.018 / 0.05`). Same four
methods, same grid, same statistics table, same two figures — so it is directly readable as
"the benchmark at that tuning". Each arm is named `<base>_normalized`, which makes the
normalization explicit for the three relative filters (whose bases do not normalize) and
for the EKF (whose base does).

The two changes are deliberately *not* separable here, and the script says so: normalizing
at a fixed std is itself a retuning, so a std chosen for raw-magnitude inputs means
something different against unit-length ones. This script answers "what does the benchmark
look like at this one tuning"; `normalization_comparison` below is where the factors are
pulled apart.

Because the std triple lives outside the method name — it is a module constant, not a
suffix — a re-tuned `mag_on_normalized` would otherwise overwrite the default-tuned one at
the same path. Everything is namespaced under a **variant** subdirectory
(`results/joint_angles/normalized_benchmark/…`) with its own statistics file, and every
manifest records the stds that produced it, so this run and the benchmark coexist. The
stds are also flags, so the same machinery runs any other tuning:

```bash
python -m experiments.normalized_benchmark                          # ~20 min on 12 cores
python -m experiments.normalized_benchmark --subjects 01 --activities walking
python -m experiments.normalized_benchmark --acc-std 0.05           # a different tuning
python -m experiments.normalized_benchmark --stats-only             # re-aggregate what is on disk
python -m plotting.normalized_benchmark                             # -> plots/normalized_benchmark/
```

The console summary prints each arm's pooled RMSE next to the shipped benchmark's number
for the same base, along with **what that file was tuned at**, read from its manifest. That
caveat is load-bearing: the benchmark is overwritten in place on re-run, so if the file on
disk predates a change to the `DEFAULT_*_STD` constants the delta spans two retunings
rather than one, and the summary flags it when it does. It writes the ~1 GB concatenated
time series only under `--save-timeseries`; the statistics table is what the figures read.

`normalization_comparison` runs every base method under three arms and prints them side
by side:

| arm | measurements | stds |
| --- | --- | --- |
| `<base>_unnormalized` | raw magnitude (identical to a bare `<base>`) | as configured |
| `<base>_normalized` | unit length | as configured |
| `<base>_rescaled` | unit length | divided by each sensor's nominal magnitude |

The third arm exists because the first two differ in *two* ways at once. Normalizing
scales a sensor's residual and Jacobian by `1/|v|` while its entry in `R` stays put, so
`normalized` also trusts a 9.81 m/s² accelerometer ~96x less relative to the gyro — a
difference between those two arms cannot say whether the geometry or the tuning moved.
`rescaled` holds the weighting fixed and changes only the geometry, which splits the
comparison cleanly:

-   `rescaled` vs `unnormalized` — the cost of discarding per-sample magnitude.
-   `normalized` vs `rescaled` — the ~96x accelerometer de-weighting on its own.

The rescaling uses a **fixed** nominal magnitude per sensor, not each sample's own `|v|`;
rescaling per sample is an exact algebraic no-op that reproduces `unnormalized` bit for
bit, since `h`, `H` and `R` would all scale together. See the `NOMINAL_*_MAGNITUDE`
comment in `experiments/experiment_utils.py`.

```bash
python -m experiments.normalization_comparison --subjects 06
```

`threshold_sensitivity` sweeps the one tuned constant MAJIC has — `DEFAULT_MAG_ADAPT_THRESHOLD`,
the `o^J` value above which the magnetometer is gated off. The grid is four thirds of a decade
either side of the shipped value in third-of-a-decade steps, **anchored on it** so the default is
a measured point and not an interpolation. Each threshold is just a parametrized method name
(`mag_adapt_th1000.00`), so the sweep reuses the vanilla pipeline's workers unchanged.

Two things come out of it. The **limits are part of the sweep**: `mag_adapt` gates on
`o^J > threshold`, so a threshold below every sample *is* `mag_off` and one above every sample
*is* `mag_on`, and both arms are run in the same invocation rather than read out of the benchmark
— the curve converging to them is a check on the pipeline, which a comparison across two runs on
two days would not be. And a **duty-cycle table**, since `(m/s²)(m/s³)` means nothing to a reader:
per trial and joint, the fraction of samples each threshold gates, computed with the identical
post-projection `o^J` the filter compares against. That is what turns the figure's x-axis into
"how often is the magnetometer switched off", and it is what makes the duty-cycle claim in the
`DEFAULT_MAG_ADAPT_THRESHOLD` comment checkable rather than quoted. It is not the smoothed,
winsorized `o^J` `sensor_distributions` tabulates for plotting.

The duty cycle is cheap and independent of the filter runs, so it can be refreshed on its own;
the full sweep is the most expensive script here (~2 h on 12 cores, 11 filter arms × 19 trials).

```bash
python -m experiments.threshold_sensitivity --gating-only     # duty cycle only, ~3 min
python -m experiments.threshold_sensitivity --subjects 06     # one subject
python -m experiments.threshold_sensitivity --stats-only      # re-aggregate what is on disk
```

`distortion_tolerance` answers the transferability question the main result leaves open. The
paper shows the magnetometer helping the proximal joints and hurting the distal ones, which is
a statement about *this lab's* floor; the sweep turns it into a **dose-response** by re-running
`mag_on` with the estimated magnetic distortion scaled to 0%, 25%, …, 200% of what was
measured, and finding where each joint's curve crosses that joint's own `mag_off`.

The dial is defined in the world frame: rotate a reading out with ground truth, scale only its
residual against the subject's assumed uniform field, rotate back
(`experiment_utils._compute_scaled_mag`). Its two ends are **exact, not approximate** — 0% *is*
the existing `perfect_mag` oracle arm and 100% *is* the ordinary real reading, both bit for bit
— so the sweep's endpoints are arms the pipeline already runs, and the `mag_on_dist1.00` vs
`mag_on` agreement is checked and printed at the end of every run. Scales above 100% amplify
the same spatial pattern, which is what lets the sweep find a breaking point that a lab with a
clean field would not reach. Each level is a parametrized method name (`mag_on_dist0.50`), so
the vanilla workers run unchanged.

Because a scale factor means nothing outside this dataset, the sweep also records **what the
dose is in degrees**, in two forms: each segment's angle against the assumed field (what an
absolute/EKF estimate suffers from) and the angle between a joint's two sensors' fields (what a
*relative* correction suffers from — distortion common to both cancels in the relative update).
The second is the axis a tolerance should be quoted on, and it is zero by construction at 0%.

```bash
python -m experiments.distortion_tolerance                       # Subject 01, ~25 min on 12 cores
python -m experiments.distortion_tolerance --distortion-only     # dose tables only, ~20 s
python -m experiments.distortion_tolerance --all                 # every subject
```

`acceleration_projection`, `sensor_distributions` and `relative_vs_absolute` do not need
Step 1: all three measure properties of the data rather than of the filter and never run the
EKF, so they can be run on the raw data alone.

`relative_vs_absolute` is the evidence behind the method's central claim — that comparing the
two sensors of a joint against *each other* beats correcting each one against a global
reference. It is a geometry result first and a measurement second, and it reports two things
whose effect sizes differ by two orders of magnitude, so the console report keeps them apart:

| what | claim | measured here |
| --- | --- | --- |
| each segment's absolute orientation | `θ_rel ≤ θ_J + θ_K` (geodesic triangle inequality) | relative is **12.6°** better at the median (mag) |
| the joint angle between them | `θ_rel ≤ θ_comp` (minimum rotation property) | relative is **0.02°** better — real, and negligible |
| the part no rotation can remove | the same inequality on the acc–mag angle β, which is rotation-invariant | relative floor **3.5°** vs **6.5°**, i.e. **1.9×** |

The middle row is a negative result and is reported as one: correcting both sensors to a shared
global reference produces two orientation errors that are largely *common mode* and cancel in
the joint angle. The exact closed form
`cos(θ_comp/2) = cos(θ_rel/2)·cos(ψ/2)` — verified to 4×10⁻¹⁴ on every sample — says why, since
it means `θ_comp ≈ √(θ_rel² + ψ²)` and the out-of-plane twist `ψ` is a few tenths of a degree
here. The joint-angle advantage instead comes from the third row, which is invariant to
rotation and therefore cannot cancel. All four identities are checked on every sample
(0/917 cells over a 10⁻⁴ deg tolerance), and the result is insensitive to the choice of global
reference, including a per-joint least-squares-optimal one.

```bash
python -m experiments.relative_vs_absolute --subjects 06   # one subject
python -m experiments.relative_vs_absolute --report-only   # reprint from what is on disk
```

`sensor_distributions` also replaces the old root-level `generate_sensor_stats.py`. It prints
the sensor-characterization report (magnetic distortion per segment, the per-subject global
field, the intrinsic noise floor from ground-anchored feet, and the child-vs-parent field
consistency behind MAJIC) and writes the per-sample distributions its figures are drawn from:

```bash
python -m experiments.sensor_distributions --subjects 06     # one subject
python -m experiments.sensor_distributions --report-only     # reprint from what is on disk
```

### Step 3: Generate the paper figures

```bash
python -m plotting.paper_figures
```

Every experiment has a matching plot script under `plotting/` with the same basename, so
each one is `python -m plotting.<experiment name>`:

```bash
python -m plotting.normalized_benchmark
python -m plotting.oracle_ablation
python -m plotting.threshold_sensitivity
python -m plotting.distortion_tolerance
python -m plotting.noise_sensitivity
python -m plotting.ekf_oracle_comparison
python -m plotting.drift_observability
python -m plotting.acceleration_projection
python -m plotting.sensor_distributions
python -m plotting.relative_vs_absolute
```

These read only from `results/` — they never re-run the filter, so a figure can be re-tuned
in seconds. If the statistics file is missing, the script says which experiment to run first.
Figures are written to `plots/`, namespaced per experiment.

`plotting.sensor_distributions` draws the distribution figures (box / ridgeline / strip per
metric), the time series through one sitting bout, and — with `--diagnostics` — a per-trial
check of both interval detectors:

```bash
python -m plotting.sensor_distributions --sides both --plot-types box
python -m plotting.sensor_distributions --example-subject 06 --diagnostics
```

`plotting.relative_vs_absolute` writes a nine-panel supplementary figure, a three-panel
headline figure for the main text, every panel again as its own file, and the blocked
significance report (`relative_vs_absolute_stats.csv`; Friedman + Wilcoxon over 44
subject × joint blocks, Holm-corrected). Its panel A is analytic — the spherical triangle the
argument is about — and panel C tests the closed form against the data with no fitted
parameter:

```bash
python -m plotting.relative_vs_absolute --composed-only
python -m plotting.relative_vs_absolute --subject 06 --activity complexTasks
```

`plotting.threshold_sensitivity` writes the four-panel gating-threshold supplement, each panel
again as its own file, and its blocked significance report
(`threshold_sensitivity_stats.csv`; Friedman across the swept thresholds + pairwise Wilcoxon,
Holm-corrected across the pooled, ankle and lumbar panels). Panel A is the duty cycle, so the
threshold axis can be read in percent-of-samples-gated; panels B and C are accuracy against
that axis, bracketed by the `mag_off`/`mag_on` limits; panel D is the per-block optimum and what
the one fixed default costs against it — which is the number behind the claim that the threshold
is not tuned:

```bash
python -m plotting.threshold_sensitivity --composed-only
```

`plotting.distortion_tolerance` writes the four-panel distortion supplement and its blocked
significance report. Panel A converts the sweep's percent axis into degrees of inter-sensor
field disagreement, so the tolerance can be quoted in a unit another lab can measure; panel B is
the pooled accuracy curve against the `mag_off`/`mag_on` references; panel C is each joint's
curve differenced against its own `mag_off`, where crossing zero *is* the tolerance; panel D puts
each block's tolerance next to the level this lab actually presents.

This figure **blocks on activity** (subject × joint × activity), unlike every other figure here.
That is deliberate and measured: the ankle's response to distortion runs the opposite way in
walking and in complex tasks, so averaging the activities into one block cancels the effect the
figure is about and reports a flat curve.

`--split-sides` additionally keeps the left and right legs apart, which is worth doing — on
Subject 01 the pooled ankle reports one crossing and the split ankle reports three, because the
two feet sit in different parts of the floor's anomaly. Note the asymmetry in what that flag
changes: the descriptive panels (A, C, D) split, while **the significance test and panel B stay
blocked on the joint type either way**. That is forced, not stylistic — the two sides correlate
at ICC ~0.5, so testing them as separate replicates would narrow every band and p-value by ~1.5×
on a precision claim the data does not support. See the `STAT_BLOCK_COLS` comment.

```bash
python -m plotting.distortion_tolerance --composed-only
python -m plotting.distortion_tolerance --split-sides   # writes *_by_side.png alongside
```

## Tests

The suite uses `unittest` from the standard library — no extra dependencies. `test/` is
not part of the installed package, so run it from the repository root:

```bash
python -m unittest discover -s test -t . -p "Test*.py"
```

It covers the toolchest (`IMUTrace`, `WorldTrace`, `PlateTrial`) and the relative filter,
including numerical checks of the EKF's measurement and noise Jacobians against finite
differences of its own residual function, plus the analysis layer on top of them:

| file | covers |
| --- | --- |
| `TestMethodSpec` | `resolve_method_spec`'s name grammar, and the round trip from every name the experiment scripts build (including that `_unnormalized` is not parsed as `_normalized`) |
| `TestExperimentPhysics` | the acc/mag oracles, the virtual EKF ground plate, and the `o^J` observability metric |
| `TestErrorStats` | `compute_error_stats` — error convention, every summary metric against numpy, grouping and the timestamp merge |
| `TestPlotUtils` | `plotting/utils.py` — the block reduction that sets *n*, Holm correction, effect sizes, and the two-stage correction family |
| `TestSensorDistributions` | the sitting/standing and ground-anchored-stationary detectors, the length-weighted noise floor, the pooled-with-margins quantile summary, and the fixed field-reference rule behind `var_reduction` |
| `TestRelativeVsAbsolute` | the spherical geometry behind the relative-correction claim: the quaternion layer against scipy, the spherical excess against a known area, all seven identities over random configurations on the whole sphere, the equality case that makes the inequalities tight, the documented counter-example the claim does *not* cover, and the rotation-invariance of the acc–mag angle |
| `TestDistortionTolerance` | the magnetic-distortion dial: that 0% *is* the mag oracle and 100% *is* the real reading, both exactly; that the residual scales linearly and amplifies past 100%; the three-way `mag_source` dispatch; and that a joint's inter-sensor dose is computed in the world frame, so two sensors in one field never disagree however differently they are oriented |
| `TestGravityConvention` | `EXPECTED_GRAVITY` against the real accelerometers |

The emphasis is on the failures that are otherwise silent — a transposed rotation, a
flipped gravity sign, a signed mean standing in for an RMSE, an uncorrected p-value
reaching a significance bracket. All of those produce output of the right shape and a
plausible magnitude, so only a value check catches them.

Most tests are synthetic and run in milliseconds. Two files touch real data —
`TestPlateTrial` loads one trial through the full `from_folder` path, and
`TestGravityConvention` checks `EXPECTED_GRAVITY` against the accelerometers — and both
skip cleanly if `data/` has not been populated, so the suite passes in a bare checkout.

# Configuration
The main plotting script, `plotting/paper_figures.py`, contains a global configuration section at the top of the file where you can easily modify the analysis and plotting parameters:

-   `SUBJECTS_TO_PLOT`: A list of subject IDs to include in the analysis.

-   `METHODS_TO_PLOT`: A list of the estimation methods to compare.

-   `METRICS_TO_PLOT`: The performance metrics to be plotted (e.g., RMSE_deg).

-   `PLOT_STYLE`: Choose between 'strip' (strip plot with median/IQR) or 'bar' (bar plot with mean/CI).

-   `SAVE_PLOTS` and `SHOW_PLOTS`: Control whether plots are saved to disk and/or displayed on screen.

## Citation

If you use the MAJIC filter in your research or wish to reference the results from our evaluation, please cite both the associated paper and this software implementation.

The citation metadata for this repository is maintained in [`CITATION.cff`](./CITATION.cff). You can export the citation in your preferred format (BibTeX, APA, etc.) by clicking the **"Cite this repository"** button in the About section of the GitHub sidebar.
