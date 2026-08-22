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
│   ├── alborno/                            # one directory per dataset
│   │   ├── Subject01/
│   │   │   ├── walking/
│   │   │   │   ├── imu data/               # Raw IMU .txt files
│   │   │   │   ├── madgwick (al borno)/    # Outputs from Al Borno et al. (2022)
│   │   │   │   └── walking.trc             # Mocap ground truth
│   │   │   └── complexTasks/               # ... same structure
│   │   └── ... (Subject02 … Subject11)
│   └── IMoveLab_Raw_Data/                  # mocap_ref/ and biplane_ref/ halves
│
├── results/                                # <-- ALL GENERATED DATA
│   ├── joint_angles/
│   │   ├── alborno/01/walking/mag_on.parquet     # <dataset>/<subject>/<trial>/<method>
│   │   ├── imove/s13/t1_walking_001/mag_on.parquet
│   │   └── <variant>/alborno/01/walking/...     # a run at non-default filter stds
│   ├── statistics/
│   │   ├── all_subject_alborno_statistics.parquet   # benchmark summary (paper figures)
│   │   ├── all_subject_alborno_joint_angles.parquet # concatenated time series, opt-in (~1 GB)
│   │   ├── oracle_ablation_statistics.parquet  # one per named experiment
│   │   └── per_subject/<experiment>/<dataset>/01/walking.parquet
│   └── experiments/
│       ├── filter_gains/<dataset>/          # the gain sweep surface each tuning is read off
│       ├── drift_observability/
│       └── global_assumptions/<dataset>/     # per-trial sample tables behind its figures
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

**The sidecar is a gate, not just a record.** Every cached layer is checked against current
code before it is read, and an artifact that does not match is refused rather than served:

| Layer | Read through | Refuses when |
| --- | --- | --- |
| `results/trials/` | `load_trial` → `StaleTrialCache` | the toolchest code, the source files or the schema moved, or the parquet is truncated |
| `results/joint_angles/` | `load_joint_angles` → `StaleJointAngles` | the filter code, the method spec, the tuning or the geometry moved; the trial underneath is stale or has been rebuilt; the parquet is truncated |
| `results/statistics/per_subject/` | `load_per_trial_statistics` → `StaleStatistics` | the constants, variant, pooled method set or requested axes differ, or the joint angles it pooled have been regenerated |

All three raise `StaleArtifact`, so every experiment inherits the refusal from the shared
loader rather than implementing its own. This exists because the silent version of it cost a
real result: Al Borno's cached angles carried `acc_std=0.018 / mag_std=0.05` while the summary
manifest claimed `0.09695 / 0.009695` — a 28x difference in magnetometer trust — and the
"mag_on and mag_off agree at every joint" that came out of it was pure artifact. The constants
were in the sidecars the whole time; nothing compared them.

The error names every stale artifact and the command that rewrites them. To look at an old
run anyway, knowing its numbers do not describe current code:

```bash
MAJIC_ALLOW_STALE=1 python -m experiments.benchmark_experiment --dataset alborno --stats-only
```

That warns loudly on every read and prints a banner to stderr. There is no other way past it,
deliberately — a fallback that costs time and nothing else is also a fallback nobody notices.

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
-   **`experiments/anatomical_frames.py`**: The rotation from each parent sensor's frame to its
    segment's anatomical frame, measured from the source mocap's landmarks and composed with the
    sensor-to-segment rotation the build recorded. It computes no error of its own — it is what
    lets `compute_error_stats` report an error along flexion / adduction / internal rotation
    instead of along the axes of a plate that was re-strapped per subject. `alborno` and `imove`
    only; the biplane halves have no registered marker set.
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
└── alborno/
    ├── Subject01/
    │   ├── walking/
    │   │   ├── walking.trc             # Mocap ground truth
    │   │   ├── imu data/               # Raw IMU .txt files, one per segment
    │   │   └── madgwick (al borno)/    # Optional: Al Borno et al. (2022) outputs
    │   └── complexTasks/               # ... same structure
    ├── Subject02/
    │   └── ...
    └── ...
```
-   Each dataset gets its own directory directly under `data/`; the Al Borno subjects go
    under `data/alborno/`, and the IMoVE download unpacks to `data/IMoveLab_Raw_Data/`.
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
python -m experiments.benchmark_experiment --dataset alborno
```

It runs one dataset at a time, over every trial BUILT under `results/trials/<dataset>/`.
`--dataset` takes `alborno`, `imove`, `imove_biplane` (fluoroscopic bone poses) or
`imove_biplane_vicon` (the same IMUs against the skin marker cluster). Everything that
differs between them — the joint table, which way is up, the magnetometer reference, whether
there is a magnetometer at all — comes from `global_assumptions.tracking_spec`. A dataset
whose IMUs have no magnetometer supports neither `mag_on` nor `mag_adapt`: left to the
default method list they are dropped with a printed note, and asked for explicitly they are
an error, because the filter would otherwise return `mag_off`'s answer under `mag_on`'s name.

This writes:

-   `results/joint_angles/<dataset>/<subject>/<trial>/<method>.parquet` — per-method joint
    angles (rotation vectors), one file per trial/method.
-   `results/statistics/per_subject/all_subject/<dataset>/<subject>/<trial>.parquet` —
    per-trial error statistics.
-   `results/statistics/all_subject_<dataset>_statistics.parquet` — the summary statistics
    used by the plotting scripts. It is the concatenation of the per-trial tables above:
    `compute_error_stats` already groups per trial, so the two are the same numbers.
-   `results/statistics/all_subject_<dataset>_joint_angles.parquet` — the concatenated
    every-sample time series. Only with `--pooled-angles`, because it is ~1 GB on alborno
    and roughly ten times that on imove, and nothing in the summary needs it.

Each of these gets a `.manifest.json` provenance sidecar alongside it.

Useful flags: `--subjects`, `--trials`, `--methods` to restrict the grid, `--workers`
to cap parallelism, and `--stats-only` to skip regeneration and re-run just the
aggregation over the joint angles already on disk. `--stats-only` is the one path that reads
every cached layer and writes none of them, so it is where the freshness gate above matters
most: it refuses rather than summarising angles that no longer match the code:

```bash
python -m experiments.benchmark_experiment --dataset alborno --subjects 01 02 --trials walking
python -m experiments.benchmark_experiment --dataset imove --subjects s13
python -m experiments.benchmark_experiment --dataset alborno --stats-only
```

#### Splitting the error by anatomical axis

By default the statistics carry one row per error `axis`: `MAG` (the rotation magnitude, what
every figure reports) plus `X`, `Y`, `Z` — the components in the **parent sensor's own frame**.
Those components are physically well defined and anatomically meaningless: the plates are
re-strapped per subject, so `X` names a different direction for each of them and pooling eleven
subjects averages flexion error into rotation error.

`experiments/anatomical_frames.py` measures the missing rotation. For each parent sensor it reads
the anatomical landmarks straight out of the source mocap (Al Borno's standing capture, which is
the only file there carrying medial markers; IMoVE's per-frame epicondyles and malleoli), builds
an ISB-shaped segment frame from them, and composes it with the sensor-to-segment rotation the
build recorded in its `*.build.parquet` sidecar:

```bash
python -m experiments.anatomical_frames --dataset alborno
python -m experiments.benchmark_experiment --dataset alborno --stats-only --anatomical-axes
python -m plotting.paper_figures --dataset alborno --axes anatomical
```

No filter is re-run — this is a statistics-stage option, so `--stats-only` is enough to add the
axes to a finished benchmark. Nothing existing is replaced either: the new `FE`, `AA` and `IE`
rows sit beside `MAG`/`X`/`Y`/`Z`, so every script that reads `axis == 'MAG'` is unaffected.
Positive means flexion, adduction and internal rotation on **both** sides of the body, which is
what makes a pooled left+right `Knee` row legitimate.

Because the basis is orthonormal, the split is exact in the strong sense that
`RMSE_MAG² = RMSE_FE² + RMSE_AA² + RMSE_IE²` in every cell — the three panels add up to the
magnitude figure and cannot contradict it. It is **not** a Cardan/Euler per-plane angle error,
which is a different quantity, is sequence-dependent and does not decompose the total.

Available on `alborno` and `imove` only; the biplane halves have no registered marker set and are
refused by name. The module's own report carries the checks that say whether to trust it —
`align_check_deg` on the build's alignment rotation, `hinge_angle_deg` comparing the landmark
flexion axis against the axis the joint is actually observed to turn about (6–7° at Al Borno's
knees), and cross-trial agreement of the basis (median 0.1–0.7°). Read
`experiments/anatomical_frames.py`'s header before quoting any of it.

### Step 2: Run any additional experiments (optional)

Each script under `experiments/` is a variation on the same pipeline and writes its own
summary to `results/statistics/<name>_statistics.parquet`:

```bash
python -m experiments.normalized_benchmark     # the benchmark grid, normalized and re-tuned
python -m experiments.oracle_ablation          # acc/mag ground-truth ablation
python -m experiments.threshold_sensitivity    # mag_adapt observability threshold sweep
python -m experiments.distortion_tolerance     # how much magnetic distortion mag_on tolerates
python -m experiments.filter_gains --dataset alborno  # measures the filter's innovation and sweeps the two free gain ratios; DATASET_STDS is read off this
python -m experiments.ekf_oracle_comparison    # EKF vs. its oracle variants
python -m experiments.drift_observability      # segment-and-reset drift diagnostic
python -m experiments.acceleration_projection --dataset alborno  # does the acc projection agree: with markers, across a joint, along a segment
python -m experiments.magnetic_projection --dataset alborno      # the same three questions for the magnetometer, plus: can an array estimate the field gradient?
python -m experiments.sensor_placement --dataset imove           # what moving the IMU along the segment costs the joint angle, and whether the projection removes it
python -m experiments.global_assumptions       # acc/mag assumption departure, static vs moving
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
(`results/joint_angles/normalized_benchmark/<dataset>/…`) with its own statistics file, and every
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
winsorized `o^J` `global_assumptions` tabulates for plotting.

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

`acceleration_projection`, `global_assumptions` and `relative_vs_absolute` do not need
Step 1: all three measure properties of the data rather than of the filter and never run the
EKF, so they can be run on the built trials alone.

`acceleration_projection` asks whether the projected accelerometer signal every `project=True`
method consumes is real, and it asks it three times because there are three references
available and each is blind to something the others see:

```bash
python -m experiments.acceleration_projection --dataset alborno
python -m experiments.acceleration_projection --dataset imove    # the only one with two sensors per segment
python -m experiments.acceleration_projection --dataset alborno --report-only
```

* **against the markers** — the projection at the joint centre against that point's mocap
  trajectory differentiated twice. The only external reference, so the only one that can catch
  an error both segments make together; its truth is mostly differentiation noise above ~5 Hz,
  which is why the analysis cutoff is 6 Hz and why the residual is an upper bound.
* **against each other** — the parent and child sensors project to one physical point, so their
  projected readings are one vector in two frames, and that identity *is* what a relative
  filter's accelerometer residual measures. No differentiated markers; blind to a common-mode
  offset error. Its magnitude channel uses no mocap orientation at all.
* **along a segment** — two sensors on one thigh or shank, projected onto each other. The truth
  is another accelerometer, so it is band-limited by the sensors instead of by the reference.
  IMoVE only, and it is where the least comfortable result lives: the two sensors disagree about
  angular velocity by 20-32%% of its magnitude, which no rigid body can do, so the "same rigid
  segment" assumption is the weakest of the three.

The three constrain each other, which is the reason all three are run rather than whichever one
reads best: section 7 of the report predicts the cross-segment disagreement from the two marker
residuals and compares it with the measured one, and neither family can compute that alone.

`magnetic_projection` asks the same three questions of the MAGNETOMETER, in the same families,
channels and scopes, so the two sets of numbers can be read side by side — it imports the
windowing and the signal helpers from `acceleration_projection` rather than reimplementing them.
The asymmetry between the two is the point. An accelerometer reading does not transport to
another point of a rigid body unchanged; a magnetometer reading is *assumed* to, because every
filter here feeds the joint centre the sensor's raw reading, which is exact only if the field is
uniform over 10–47 cm. Nothing else in the repo checks that: `global_assumptions` asks whether
each sensor sees one constant field, which a smoothly varying room would fail while transporting
perfectly over 10 cm.

```bash
python -m experiments.magnetic_projection --dataset alborno
python -m experiments.magnetic_projection --dataset imove    # three magnetometers per thigh and shank
python -m experiments.magnetic_projection --dataset alborno --report-only
```

At the 0th order the two segments of a joint disagree about the field direction by a median
**7.6°** on IMoVE and **8.0°** on Al Borno, worst distally (R_Ankle 15.2°, hips 3–5°), and that
is the residual the relative filter's magnetometer update consumes. Its **norm channel needs no
mocap at all** — `|R m| = |m|`, so neither the orientations nor the offsets enter — which is a
stronger mocap-free claim than the accelerometer version can make, and it puts a floor under the
rest: two sensors disagreeing about field *magnitude* by 7% cannot be reconciled by any
orientation estimate.

The question the file was built for is whether several magnetometers can estimate the field
**gradient** and beat that. Eight projections are scored — the reading, the segment mean, a 1st-
and 2nd-order fit along the segment's sensor array, a per-sample and a per-trial whole-body
gradient tensor, the same tensor constrained *symmetric and traceless* as `curl B = 0` and
`div B = 0` require, and a `body_model` control that reports a fitted model and ignores the
measurement. **None of them beats the 0th order.** Three measurements say why, and they agree:

| test | what a real field gradient would give | measured |
| --- | --- | --- |
| disagreement vs sensor separation | proportional to distance | 5.5°/100 mm within a segment vs 2.2° across a joint — 2.5× *steeper* at short range |
| what explains a same-segment difference | a world-frame gradient | a constant in the **sensor's own frame** explains 85% with 3 parameters; a physically admissible gradient explains 79% with 5 |
| fitted gradient magnitude | the one the markers can see (0.45 a.u./m) | per-sample fit 0.87, i.e. 2× — it is absorbing sensor error, and only the physically constrained per-trial fit (0.38) agrees with the markers |

So the disagreement is a per-sensor, body-fixed term, not a field, and extrapolating a fitted
"gradient" past the end of a 17 cm array to a joint centre 25 cm away amplifies it. The held-out
test is the cleanest statement of the limit: two magnetometers predict the third about as well by
their **mean** as by the line through them where the target lies *between* them (ratio 0.998),
and the line loses every time it has to reach beyond them (ratio 1.10) — and the joint centre is
always beyond them. What does help is a three-parameter hard iron: 7.6° → 5.4°, against 1.3° for
the best spatial model. The caveat is reported in the same section — the fitted offset scatters
between a subject's own trials by ~92% of its own size, so most of what it removes was the room
rather than the device, and every calibrated arm here is fitted leave-one-trial-out for that
reason (the in-sample arm reads 2.5° better, which is exactly the optimism being avoided).

`body_model` is in the table because segment-to-segment agreement can be driven to **exactly
zero** by discarding the measurement, so every mode is scored on two axes — agreement, and
distance from a marker-supported field model — and the control makes that visible rather than
arguable.

The last figure asks the complementary question — not how far two magnetometers **disagree** but
how much of what they see is **shared**, which is what decides whether a relative filter's
subtraction removes a disturbance or manufactures one. Each sensor's time-mean is removed first,
so this is the fluctuation rather than the standing offset the rest of the experiment measures,
and the far pair classes are the control: a disturbance that is genuinely spatial is shared by
neighbours and not by a sensor on the other leg.

| pair | IMoVE ρ | IMoVE surviving | Al Borno ρ | Al Borno surviving |
| --- | --- | --- | --- | --- |
| same segment (7-19 cm) | 0.67 | 0.37 | — | — |
| across a joint (25-47 cm) | 0.38 | 0.72 | 0.87 | 0.29 |
| other leg | 0.33 | 0.70 | 0.72 | 0.28 |
| unrelated | 0.21 | 0.87 | 0.60 | 0.69 |

`surviving` is the fraction of disturbance energy left after subtracting the two sensors — 0 means
perfectly shared and differencing removes it, 1 means independent so differencing is neutral, above
1 means differencing makes it worse. Correlation falls off with adjacency on both datasets, so part
of the disturbance is genuinely spatial, and removing each sensor's own fitted body-fixed bias
*raises* the same-segment correlation (0.67 → 0.80), so the bias decorrelates neighbours rather
than coupling them. Split by timescale, the slow band (< 0.5 Hz) carries nearly all of the shared
part while the gait band is shared only *within* a segment — a room-scale disturbance changes as
the subject crosses the lab, whereas two biases rotating together only look alike on sensors that
rotate together.

**The two datasets disagree on how much differencing buys, and the difference is the protocol.**
Across a joint, Al Borno leaves 29% of the disturbance after subtraction and IMoVE leaves 72%. Al
Borno's trials are minutes of walking across a magnetically inhomogeneous capture volume, so the
shared slow term dominates; most of IMoVE's are short tasks performed on one spot, where there is
little room-scale variation to share and the per-sensor term is most of what is left. So the
benefit a relative formulation gets from common-mode rejection is a property of the protocol, not
a constant — which also means it is largest exactly where magnetic distortion is worst.

`sensor_placement` is the only experiment here that can vary WHERE the IMU sits, and it is IMoVE
only: three sensors on each thigh and shank, High/Mid/Low, 7-19 cm apart against a single marker
cluster. One trial, one joint, one filter, one reference construction — the only thing that
changes between two cells is which sensor on the same rigid segment the filter was handed, so a
difference between cells is placement and cannot be subject, motion or tuning. The full cross
product is run (9 cells at each knee, 3 at each hip and ankle), crossed with three projections
and the magnetometer on and off.

```bash
python -m experiments.sensor_placement --dataset imove
python -m experiments.sensor_placement --dataset imove --report-only
```

Over 21 subjects and 231 trials, moving the sensor is worth more than most of the method choices
this repo benchmarks. Within a trial and joint, the gap between the best and the worst placement
is a median **41.7°** unprojected, and error tracks the lever arm almost perfectly (Spearman
**1.00** within a cell, **16.0° per 100 mm**). MAJIC's acceleration projection is exactly the
correction for that term, and it works: with a marker joint centre the same gap falls to
**10.3°** and the slope to **1.9° per 100 mm**. With the joint centre estimated from the IMUs
alone — the deployable case — it lands at **13.1°**, so most but not all of the benefit survives
losing the markers, and what is lost is a tail rather than a level shift (median cost +0.13°, p90
+19°: the cells where the IMU-only fit failed are the distal placements, which are also the ones
that most needed the correction).

The mechanism is settled by a test that needs no extra data. A thigh sensor's distance to the hip
and its distance to the knee move in OPPOSITE directions as it slides down the segment, so a
lever-arm explanation predicts the placement ranking REVERSES between the two joints a segment
spans, while every rival explanation (that sensor is noisier, that spot has more soft tissue,
that tape job was worse) predicts the same ranking at both. Measured: the ranking reverses in
**66%** of segment-trials unprojected and **28%** once projected, against a geometry control that
reverses 100% of the time. The report also shows it directly — the best placement for the hip is
the thigh's High sensor (91% of trials), for the ankle it is the shank's Low sensor (83%), and
the worst knee cell is High-Low, both sensors as far from the knee as the segments allow.

Two secondary results. Placement MISMATCH between the two sensors costs nothing beyond the two
lever arms it implies (+0.4° at most, not significant), so there is no reason to insist both
sensors sit at the same height — only that each sits near its joint. And the magnetometer pays
off better the further the sensor is from the floor (rho 0.25-0.35 with sensor height), which is
the sign the floor-source distortion finding predicts, though the effect is small next to the
lever arm.

The confounds are measured rather than argued, in section 7 of the report: the placements of one
segment read the same |omega| to 2.5°/s (3.8% of signal), as a rigid body requires; the taped
High/Low sensors fall back to a default cluster-to-IMU offset on 8-14% of plates against 0% for
the bolted Mid ones; and their sensor-to-segment alignment residual is roughly twice the Mid
sensors' (0.25 vs 0.15 of their own gyro signal). That last one matters most once the lever arm
is gone — with the projection on, the alignment residual becomes the strongest remaining
correlate of the error (rho 0.38), which is where the residual 10° of placement spread lives.

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

`global_assumptions` measures how far the two assumptions every orientation filter here rests
on are from true — the accelerometer reads gravity and nothing else, the magnetometer reads one
constant field — and, the point of the experiment, splits every number by whether the sensor was
moving. It supersedes `sensor_distributions`, which in turn replaced the root-level
`generate_sensor_stats.py`.

A sensor is **static** wherever its gyroscope stays under 0.05 rad/s across a 0.25 s window.
Gyro-only, deliberately: a detector that consulted the accelerometer or the magnetometer would
define its own answer into existence. Its one blind spot is pure translation, and that is
measured rather than waved away — the report carries the mocap linear and angular speed observed
during every detected-static stretch (0.5–5 mm/s across both datasets). Two regimes come out of
it: `static` (this sensor still, so a stance-phase foot counts mid-walk) and `body_static`
(every sensor still at once), plus `nonstatic` and `all`. Both static regimes get their own
intrinsic noise floor, and the gap between them is the vibration a stance foot still carries.

`magdev` is reported against **four reference arms**, because "deviation from the global field"
is only as meaningful as the constant you subtract, and the obvious choice is not neutral. Every
arm is the identical computation `|m_world − c|`; only `c` differs, so any gap between two of them
is the reference and nothing else. They are declared once in `MAGDEV_ARMS`:

| arm | reference | what it is for |
| --- | --- | --- |
| `magdev` | the SUBJECT's field, pooled over all its (trial, sensor) medians | what a filter calibrated once per session suffers, and the arm the "one constant field" claim is about |
| `magdev_trial` | the same reduction over ONE trial's sensors | removes between-trial drift, leaves within-trial structure |
| `magdev_loo` | the subject arm **with this sensor left out** | the only arm unbiased by self-reference — quote this when asking how far a sensor is from a field it did not help define |
| `magdev_clean` | the single cleanest sensor (`DatasetSpec.clean_sensor`) | a **diagnostic**, matching the construction `_compute_expected_mag_field` hands the EKF's mag oracle |

Two of these carry warnings the report prints in full. `magdev_trial` is fitted on the very
samples it scores, so it is biased low and more so on short trials — and refitting the constant
per trial concedes the "one constant field" assumption at the between-trial scale, which is
evidence, not noise. `magdev_clean` **must never be the arm a proximal-to-distal gradient is
quoted from**: referencing everything against the cleanest sensor collapses that sensor's own row
toward its within-trial variation alone (the Al Borno torso goes 1.70° against the pooled
reference to 0.50° against itself), inflating the gradient from 7.1× to 26.4× while the distal end
barely moves. That is the top of the gradient being defined to zero.

`magdev_loo` is the one that corrects a real bias in the shipped numbers: a sensor is 1/N of its
own subject reference, and the sensor set is spatially lopsided (Al Borno carries 6 lower-limb
sensors near the floor against 2 upper). Leaving the sensor out moves the proximal segments up in
both datasets — Al Borno torso +0.84°, pelvis +0.43°; IMoVE pelvis +0.43° — and that rise is the
bias, concentrated proximally because that is where a sensor sits nearest the pooled reference it
helped define. Its `clean_sensor` is declared per dataset rather than chosen as the measured
argmin, which costs nothing (the argmin is the torso on 10 of 11 Al Borno subjects and the pelvis
on 23 of 26 IMoVE sessions) and avoids selecting on a statistic correlated with the measurement.

Section 4 prints all four side by side with a `loo/subj` column, and `sensor_stats` carries
`reference_drift_<arm>_deg`, the angle between each arm's constant and the subject one. The
subject-scope arms are rolled up once into `<subject>/subject_references.parquet` (long-form over
`arm × sensor`, since `loo` is a vector per sensor); the trial arm is recomputed per trial.

Eleven report sections: the headline static-vs-moving contrast, static coverage and detector
validation, the accelerometer departure, the magnetometer departure, the noise floor, joint
observability `o^J` by regime against the `mag_adapt` gate, **sensor placement vs observability**,
local field consistency, the between-sensor residuals the relative filter actually absorbs, and
per-subject and per-trial spreads. Metrics come in mocap-referenced and reference-free pairs (`linacc`/`acc_norm_dev`,
`magdev`/`mag_norm_dev`), which matters because the Al Borno walking trials open with a long
standing pause *before* the cameras start: their static regime is real and large but invisible
to anything needing a rotation.

Runs on **every dataset in the repository**. Every segment map, joint table and sensor list is a
`DatasetSpec`, so Al Borno's 8 sensors, IMoVE's 15 (three per thigh and shank, no torso) and the
biplane half's 4 go through one code path. Four spec names over three build trees:

| `--dataset` | build tree | sensors | ground truth | magnetometer |
| --- | --- | --- | --- | --- |
| `alborno` | `alborno` | 8 | marker clusters | yes |
| `imove` | `imove` | 15 | marker clusters | yes |
| `imove_biplane` | `imove_biplane` | 4 (one knee per trial) | fluoroscopic bone pose, ~0.48 s | **no** |
| `imove_biplane_vicon` | `imove_biplane` | 4 | marker clusters, ~6.7 s | **no** |

The two biplane specs analyse one build under two different ground-truth references, so they are
separate names writing to separate directories rather than one spec carrying both — the IMU is the
same device in each, and pooling them would double-count every sample. Their reference-free half
(the `|acc|` departure, the gyro-only static detector, the noise floor) is identical by
construction, which makes their disagreement on the mocap-referenced half a clean read on what the
ground-truth choice costs.

**No magnetometer is a first-class case, not a degenerate one.** An MC10 BioStamp measures
acceleration and rotation only, and the reader fills `mag` with exact zeros so nothing reads
uninitialized memory. Zeros are not a missing value: pooled into a subject field they give the
zero vector, every `magdev` against it is exactly 0, and the report would then claim MAG=CONSTANT
holds perfectly on a dataset that never measured a field. `DatasetSpec.has_magnetometer` therefore
**omits** every magnetic column rather than NaN-filling it, skips stage one entirely, and prints
sections 4 and 8 as one line saying why. A downstream slice for `magdev` gets a `KeyError` instead
of a number that means nothing.

Trials are enumerated from `results/trials/<dataset>/`, so what it analyses is exactly what has
been built — minus the **orphans**, parquets the dataset's source no longer enumerates. The build
tree is a cache and nothing prunes it, so a trial built once and later excluded leaves its file
behind: IMoVE has 25 (the `t0_static_pose` recordings, excluded by `sources.UNSYNCABLE_TRIALS`
because a static pose has no motion for the gyro cross-correlation to sync on) and `imove_biplane`
has one (`12/Test1/A/Rstatic1`). They are stale against the current toolchest digest and
`build_trials` will never refresh them, so each was a permanent `Failed (stale)` cell with no
command that could clear it. They are now skipped and counted at the top of the run.

IMoVE's three sensors per thigh and shank all sit on one segment against one marker cluster, so
they border the same joints, and the build gives each its own lever arm to the joint center (it
shifts every plate's mocap origin onto its own IMU). That makes 18 joint pairs rather than 6 —
the six Mid-to-Mid ones plus a placement-matched High and Low variant of each — and it exposes an
effect no other dataset here can measure: how `o^J` depends on where ALONG a segment a sensor is
mounted. Pooled, the answer is that it barely does — 1.07x between the extremes, against the 3-14x
that separates one anatomical joint from another — but the SIGN is a clean check on the mechanism.
o^J scales with the sensor's lever arm to the joint center, and which placement has the longer arm
depends on which end of the segment the joint is at (a thigh sensor's solved knee-center offsets
run 257 / 159 / 93 mm going down the segment, so High is furthest from the knee and Low is
furthest from the hip). That predicts High winning at the knees and ankles and Low at the hips,
and it holds on all six. The six Mid pairs are `primary_joints` and are what the by-joint tables
and figures show; all 18 are measured and saved, and section 7 is the placement comparison.

```bash
python -m experiments.global_assumptions --dataset alborno              # ~1.5 min, 19 trials
python -m experiments.global_assumptions --dataset imove                # 236 trials
python -m experiments.global_assumptions --dataset imove_biplane        # 379 trials
python -m experiments.global_assumptions --dataset imove_biplane_vicon  # the same 379, Vicon ref
python -m experiments.global_assumptions --dataset imove --subjects s13   # one session
python -m experiments.global_assumptions --dataset alborno --report-only  # reprint from disk
```

The IMoVE `t0_static_pose` recordings are excluded because they cannot be **built**: the sync
step cross-correlates gyros to find the mocap offset, and a static pose has no motion to
correlate. They would otherwise be the cleanest at-rest data in either dataset — and note that
this experiment does not actually need the correspondence for most of what it measures, since
every reference-free metric (`acc_norm_dev`, `mag_norm_dev`, the gyro-only static detector, the
whole noise floor) is computed on the raw trace and never consults a rotation. Recovering them
means a build path that emits a `PlateTrial` with `valid` all False instead of refusing to sync.

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
python -m plotting.filter_gains
python -m plotting.ekf_oracle_comparison
python -m plotting.drift_observability
python -m plotting.acceleration_projection --dataset alborno
python -m plotting.magnetic_projection --dataset alborno
python -m plotting.sensor_placement --dataset imove
python -m plotting.global_assumptions
python -m plotting.relative_vs_absolute
```

These read only from `results/` — they never re-run the filter, so a figure can be re-tuned
in seconds. If the statistics file is missing, the script says which experiment to run first.
Figures are written to `plots/`, namespaced per experiment.

`plotting.paper_figures --axes anatomical` draws the same two figures split by anatomical axis
instead of pooled into a rotation magnitude: `figure_1_rmse_by_axis.png` (one panel per axis,
sharing a y scale, because the panels are components of one vector and their relative height is
the result) and one `heatmap_rmse_<axis>.png` per axis. It needs statistics written with
`--anatomical-axes` — see [Splitting the error by anatomical axis](#splitting-the-error-by-anatomical-axis)
— and says which command to run if they are not there rather than falling back to the magnitude
under an anatomical title.

`plotting.global_assumptions` draws the headline two-panel static-vs-moving figure, the
distribution figures (regime-split box / ridgeline / strip per metric), the noise floor against
the filter's tuned stds, static coverage and detector validation, a subject x segment heatmap,
per-trial medians, a time series through a representative whole-body-static bout, and — with
`--diagnostics` — a per-trial static-detector trace: 22 figures per dataset. It takes `--dataset`
like the experiment does, including the two biplane specs, where it draws the accelerometer
figures and a two-panel noise floor and skips the magnetic ones rather than plotting empty axes.
Box statistics are computed from the full sample tables and drawn with `bxp`, so no box is
estimated off a subsample.

Observability is deliberately **not** plotted here. `o^J` is a property of a joint pair and of
the joint-center projection rather than of the global sensor assumptions this experiment measures,
so its figures belong with the joint-offset work; `scratch/observability_plots_for_joint_offset.py`
holds the ones lifted out of this module. The experiment still writes `joint_samples` and
`joint_stats` and report sections 6 and 7 still print off them — only the plotting moved.

```bash
python -m plotting.global_assumptions --dataset alborno
python -m plotting.global_assumptions --dataset imove --sides both --plot-types box
python -m plotting.global_assumptions --dataset imove_biplane
python -m plotting.global_assumptions --dataset alborno --diagnostics
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
| `TestAnatomicalFrames` | the anatomical basis and the split it produces: that positive is flexion/adduction/internal rotation on *both* sides (checked on a synthetic mirrored subject, because getting it wrong cancels the two legs in every signed statistic while RMSE looks fine), that the components are `Aᵀe` and not `Ae`, that the three axes sum to the magnitude in quadrature, and that a joint with no basis gets no anatomical rows rather than an identity |
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
