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

## Repository Structure

The repository is organized into several key Python scripts and a data directory:
```
.
├── data/
│   ├── Subject01/
│   │   ├── walking/
│   │   │   ├── imu data/                # Raw IMU .txt files
│   │   │   ├── madgwick (al borno)      # Outputs from Al Borno et. al (2022)
│   │   │   ├── walking.trc              # Mocap ground truth
│   │   │   ├── myIMUMappings_walking.xml  # Maps IMU IDs to segments
│   │   │   └── walking_orientations_mag_on.sto  # <-- Generated orientation file
│   │   └── complexTasks/
│   │       └── ... (similar structure)
│   ├── ... (data for other subjects)
│   │
│   ├── all_subject_data.pkl             # <-- Generated full time-series data
│   ├── all_subject_statistics.pkl       # <-- Generated summary statistics
│   └── all_subject_pearson_correlation.pkl # <-- Generated correlation data
│
├── plots/                                 # <-- Output directory for figures
│
├── src/
│   ├── toolchest/
│   │   ├── IMUTrace.py
│   │   ├── WorldTrace.py
│   │   ├── PlateTrial.py
│   │   └── AHRSFilter.py
│   └── RelativeFilterPlus.py
│
├── generate_method_orientation_sto_files.py
├── generate_method_data_and_stats_pkls.py
├── plot_paper_figures.py
├── plot_imu_data_in_world_frame.py
└── README.md
```
-   **`src/RelativeFilterPlus.py`**: The core implementation of the Relative Filter.
-   **`src/toolchest/`**: A collection of utility classes for handling IMU data (`IMUTrace`), motion capture data (`WorldTrace`), and synchronized trial data (`PlateTrial`).
-   **`generate_method_orientation_sto_files.py`**: This script processes the raw data for each subject and trial, runs various orientation estimation methods (including the Relative Filter variants), and saves the resulting segment orientations as `.sto` files.
-   **`generate_method_data_and_stats_pkls.py`**: This script loads the generated `.sto` files, calculates joint kinematics, computes error metrics against the "Marker" ground truth, and saves the comprehensive time-series data and summary statistics into `.pkl` files for efficient access.
-   **`plot_paper_figures.py`**: The main script for generating the statistical comparison plots presented in the paper. It loads the `all_subject_statistics.pkl` file and creates detailed figures comparing the different estimation methods.
-   **`plot_imu_data_in_world_frame.py`**: A script to visualize the raw IMU data in the world frame, useful for initial data exploration and validation.
-   **`data/`**: This directory is intended to hold the input data and the generated `.pkl` files.
-   **`plots/`**: The default output directory for all generated figures.

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

3.  **Install the required libraries:** The necessary packages are listed in `requirements.txt`.
    ```bash
    pip install numpy pandas matplotlib seaborn scipy
    ```

## Data Setup

The scripts expect a specific directory structure for the input data. You will need to populate the `data/data/` directory as follows:
```
data/
├── Subject01/
│ ├── walking/
│ │ ├── subject01_walking.trc
│ │ └── imu data/
│ │ └── ... (IMU .txt files)
│ └── complexTasks/
│ ├── subject01_complextasks.trc
│ └── imu data/
│ └── ... (IMU .txt files)
├── Subject02/
│ └── ...
└── ...
```
-   Each subject should have their own directory (e.g., `Subject01`, `Subject02`).
-   Inside each subject's directory, there should be subdirectories for each trial type (e.g., `walking`, `complexTasks`).
-   Each trial directory must contain:
    -   A motion capture file in `.trc` format.
    -   An `imu data` subdirectory containing the raw IMU data in `.txt` format for each segment.

## Usage and Workflow

To reproduce the results from the publication, run the scripts in the following order.

### Step 1: Generate Segment Orientation `.sto` Files

This step processes the raw data and runs the different orientation estimation algorithms.

```bash
python generate_method_orientation_sto_files.py
```
This will create .sto files for each method within each subject's trial directory (e.g., data/data/Subject01/walking/walking_orientations_mag_on.sto).

### Step 2: Generate Data and Statistics .pkl Files
This step calculates the joint angles from the .sto files and computes detailed error statistics, saving them in convenient .pkl files.
```bash
python generate_method_data_and_stats_pkls.py
```
This will produce three key files in the data/data/ directory:

-   `all_subject_data.pkl`: A large file containing the full time-series data for all joints, methods, and subjects.

-   `all_subject_statistics.pkl` and `all_subject_statistics.csv`: A summary file with aggregated statistics (RMSE, MAE, etc.) used for plotting.

-   `all_subject_pearson_correlation.pkl` and `all_subject_pearson_correlation.csv`: A file containing Pearson correlation results.


Note: You can set REGENERATE_FILES = False in this script to load existing .pkl files and avoid reprocessing all the data.

### Step 3: Generate the Paper Figures
This is the final step to generate the plots shown in the paper.
```bash
python plot_paper_figures.py
```
The output plots will be saved in the plots/ directory by default.

# Configuration
The main plotting script, plot_paper_figures.py, contains a global configuration section at the top of the file where you can easily modify the analysis and plotting parameters:

-   `SUBJECTS_TO_PLOT`: A list of subject IDs to include in the analysis.

-   `METHODS_TO_PLOT`: A list of the estimation methods to compare.

-   `METRICS_TO_PLOT`: The performance metrics to be plotted (e.g., RMSE_deg).

-   `PLOT_STYLE`: Choose between 'strip' (strip plot with median/IQR) or 'bar' (bar plot with mean/CI).

-   `SAVE_PLOTS` and `SHOW_PLOTS`: Control whether plots are saved to disk and/or displayed on screen.

# Citation
If you use this code or the Relative Filter in your research, please cite our publication:

[Your Publication Citation Here - e.g., Author(s), "Title of Paper," Journal, Volume, Pages, Year.]
