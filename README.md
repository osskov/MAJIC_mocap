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
│   │   └── Subject01/walking/mag_on.parquet
│   ├── statistics/
│   │   ├── all_subject_statistics.parquet      # benchmark summary (paper figures)
│   │   ├── all_subject_joint_angles.parquet    # concatenated time series (~1 GB)
│   │   ├── oracle_ablation_statistics.parquet  # one per named experiment
│   │   └── per_subject/<experiment>/Subject01/walking.parquet
│   └── experiments/
│       ├── noise_sensitivity/
│       └── drift_observability/
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
python -m experiments.oracle_ablation          # acc/mag ground-truth ablation
python -m experiments.threshold_sensitivity    # mag_adapt observability threshold sweep
python -m experiments.noise_sensitivity        # gyro/acc/mag noise parameter sweep
python -m experiments.ekf_oracle_comparison    # EKF vs. its oracle variants
python -m experiments.drift_observability      # segment-and-reset drift diagnostic
```

### Step 3: Generate the paper figures

```bash
python -m plotting.paper_figures
```

Every experiment has a matching plot script under `plotting/` with the same basename, so
each one is `python -m plotting.<experiment name>`:

```bash
python -m plotting.oracle_ablation
python -m plotting.threshold_sensitivity
python -m plotting.noise_sensitivity
python -m plotting.ekf_oracle_comparison
python -m plotting.drift_observability
```

These read only from `results/statistics/` — they never re-run the filter, so a figure
can be re-tuned in seconds. If the statistics file is missing, the script says which
experiment to run first. Figures are written to `plots/`, namespaced per experiment.

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
| `TestMethodSpec` | `resolve_method_spec`'s name grammar, and the round trip from every name the experiment scripts build |
| `TestExperimentPhysics` | the acc/mag oracles, the virtual EKF ground plate, and the `o^J` observability metric |
| `TestErrorStats` | `compute_error_stats` — error convention, every summary metric against numpy, grouping and the timestamp merge |
| `TestPlotUtils` | `plotting/utils.py` — the block reduction that sets *n*, Holm correction, effect sizes, and the two-stage correction family |
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
