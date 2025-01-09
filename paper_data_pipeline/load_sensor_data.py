import os.path
from typing import List, Dict

import numpy as np
from scipy.stats import wilcoxon
from src.IMUSkeleton import IMUSkeleton, RawReading
from src.toolchest.PlateTrial import PlateTrial
from paper_data_pipeline.generate_kinematics_from_precomputed_plates import generate_kinematics_report
import nimblephysics as nimble
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import re

plt.rcParams.update({
    'font.size': 20,  # Increase font size for all text
    'font.family': 'sans-serif',  # Use a sans-serif font family
    'font.sans-serif': ['Arial'],  # Set Arial as the default sans-serif font
    'axes.titlesize': 22,  # Larger title font size
    'axes.labelsize': 18,  # Larger axis label font size
    'xtick.labelsize': 18,  # Larger x-axis tick font size
    'ytick.labelsize': 18,  # Larger y-axis tick font size
    'legend.fontsize': 16,  # Larger legend font size
    'axes.labelweight': 'bold',  # Make axis labels bold
    'axes.titleweight': 'bold',  # Make titles bold
})
method_display_names = {
    'Mag Free': 'Magnetometer Free',
    'Never Project': 'MAJIC Zeroth Order',
    'Always Project': 'MAJIC First Order',
    'Cascade': 'MAJIC Adaptive'
}
method_colors = {
    method: color for method, color in zip(list(method_display_names.keys()), plt.cm.tab10.colors)
}


load_sensor_stats = True
generate_sensor_stats = False

if generate_sensor_stats:

    subjects_projected_readings: List[Dict[str, Dict[
        str, List[
            np.array]]]] = []  # List by subject, of dicts with joint names as keys and sensor data lists as values

    # Collect kinematics
    for subject_number in range(1, 4):
        subject_skeleton = IMUSkeleton.load_from_json(f"../data/S{subject_number}_IMU_Data/Walking/skeleton.json")
        plate_trials = PlateTrial.load_cheeseburger_trial_from_folder(f"../data/S{subject_number}_IMU_Data/Walking/")

        all_readings_list: List[Dict[str, RawReading]] = RawReading.create_readings_from_plate_trials(plate_trials)
        all_readings_list = all_readings_list[5000:48000]

        projected_reading_list = {joint_name: {'parent_acc': [], 'parent_avg_mag': [], 'parent_proj_mag': [],
                                               'child_acc': [], 'child_avg_mag': [], 'child_proj_mag': []} for
                                  joint_name in
                                  subject_skeleton.joints.keys()}  # Each joint will have a list of data for each type

        for i, readings_dict in enumerate(all_readings_list):
            for joint_name, joint in subject_skeleton.joints.items():
                # Create arrays to hold our projected acc, averaged mag and projected mag for the parent and child
                joint.joint_filter.readings_projector.mag_choosing_method = 'project'

                # Find and project the readings for the parent and child
                raw_parent_reading = next(
                    readings_dict[parent_name] for parent_name in readings_dict if
                    parent_name.__contains__(joint.parent_name))
                raw_child_reading = next(
                    readings_dict[child_name] for child_name in readings_dict if
                    child_name.__contains__(joint.child_name))

                parent_reading, child_reading = joint.joint_filter.readings_projector.get_projected_readings_at_joint_center(
                    parent_reading=raw_parent_reading, child_reading=raw_child_reading)

                projected_reading_list[joint_name]['parent_acc'].append(parent_reading.acc)
                projected_reading_list[joint_name]['parent_avg_mag'].append(
                    np.mean(np.array(raw_parent_reading.mags), axis=0))
                projected_reading_list[joint_name]['parent_proj_mag'].append(parent_reading.mag)
                projected_reading_list[joint_name]['child_acc'].append(child_reading.acc)
                projected_reading_list[joint_name]['child_avg_mag'].append(
                    np.mean(np.array(raw_child_reading.mags), axis=0))
                projected_reading_list[joint_name]['child_proj_mag'].append(child_reading.mag)

        subjects_projected_readings.append(projected_reading_list)

    # Print mean and std for each joint sensor type
    for subject_number, subject_readings in enumerate(subjects_projected_readings):
        for joint_name, readings in subject_readings.items():
            parent_mags = np.array(readings['parent_avg_mag'])
            child_mags = np.array(readings['child_avg_mag'])
            parent_proj_mags = np.array(readings['parent_proj_mag'])
            child_proj_mags = np.array(readings['child_proj_mag'])
            parent_accs = np.array(readings['parent_acc'])
            parent_acc_diff = np.diff(parent_accs, axis=0)
            parent_obs = np.cross(parent_acc_diff, parent_accs[:-1]) / np.linalg.norm(parent_accs, axis=1)[:-1][:, np.newaxis]
            child_accs = np.array(readings['child_acc'])
            child_acc_diff = np.diff(child_accs, axis=0)
            child_obs = np.cross(child_acc_diff, child_accs[:-1]) / np.linalg.norm(child_accs, axis=1)[:-1][:, np.newaxis]


            parent_mags_norms = np.array([np.linalg.norm(mag) for mag in parent_mags])
            child_mags_norms = np.array([np.linalg.norm(mag) for mag in child_mags])
            parent_proj_mags_norms = np.array([np.linalg.norm(mag) for mag in parent_proj_mags])
            child_proj_mags_norms = np.array([np.linalg.norm(mag) for mag in child_proj_mags])
            parent_acc_norms = np.array([np.linalg.norm(acc) for acc in parent_accs])
            child_acc_norms = np.array([np.linalg.norm(acc) for acc in child_accs])
            parent_obs_norms = np.array([np.linalg.norm(obs) for obs in parent_obs])
            child_obs_norms = np.array([np.linalg.norm(obs) for obs in child_obs])


            print(f"Subject {subject_number + 1} - {joint_name}")
            print(f"Parent Mag Norms: Mean: {np.mean(parent_mags_norms)}, Std: {np.std(parent_mags_norms)}")
            print(f"Child Mag Norms: Mean: {np.mean(child_mags_norms)}, Std: {np.std(child_mags_norms)}")
            print(f"Parent Proj Mag Norms: Mean: {np.mean(parent_proj_mags_norms)}, Std: {np.std(parent_proj_mags_norms)}")
            print(f"Child Proj Mag Norms: Mean: {np.mean(child_proj_mags_norms)}, Std: {np.std(child_proj_mags_norms)}")
            print(f"Parent Acc Norms: Mean: {np.mean(parent_acc_norms)}, Std: {np.std(parent_acc_norms)}")
            print(f"Child Acc Norms: Mean: {np.mean(child_acc_norms)}, Std: {np.std(child_acc_norms)}")
            print(f"Parent Obs Norms: Mean: {np.mean(parent_obs_norms)}, Std: {np.std(parent_obs_norms)}")
            print(f"Child Obs Norms: Mean: {np.mean(child_obs_norms)}, Std: {np.std(child_obs_norms)}")
            print()

if load_sensor_stats:
    subjects_errors = []
    # Collect kinematics
    for subject_number in range(1, 4):
        subject_error = generate_kinematics_report(
            b3d_path=f"../data/S{subject_number}_IMU_Data/S{subject_number}.b3d",
            model_path=f"../data/S{subject_number}_IMU_Data/SimplifiedIMUModel.osim",
            cached_history_path=f"../data/S{subject_number}_IMU_Data/Walking/cached_skeleton_history.npz",
            output_path=f"../data/S{subject_number}_IMU_Data/Walking/",
            output_csv=f"../data/Walking_Results/S{subject_number}_Walking_joint_angles.csv"
        )
        subject_error = {joint_name: {method: method_data[5000:48000] for method, method_data in joint_data.items()} for
                         joint_name, joint_data in subject_error.items()}
        subjects_errors.append(subject_error)

    data_file = "../data/Walking_Results/sensor_data_distributions.txt"
    # Read the text file
    with open(data_file, 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store the data
    data = []

    # Parse the lines and extract values
    for line in lines:
        # Extract Subject, Measure and Norms
        match = re.match(r"Subject (\d+) - (\w+)", line)
        if match:
            subject, measure = match.groups()
            continue

        # Extract norms
        norm_match = re.match(r"(.*) Norms: Mean: ([\d\.]+), Std: ([\d\.]+)", line)
        if norm_match:
            norm_type, mean, std = norm_match.groups()
            data.append([subject, measure, norm_type, float(mean), float(std)])

    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data, columns=['Subject', 'Joint', 'Sensor', 'Mean', 'Std'])

    # Display the DataFrame
    print(df)


    def generate_acc_std_vs_error_scatter_plot():
        # Assuming method error is the 'Mean' column, if not, adjust accordingly
        plt.figure(figsize=(10, 6))
        axis_shape = ['o', 's', '^']

        correlation_child_acc_std = []
        correlation_method_errors = {method: [] for method in method_display_names.keys()}

        for subject_number, subject_error in enumerate(subjects_errors):  # Iterate over the subjects
            for joint_name in subject_error:
                # First, isolate the parent acc stds from the df
                parent_acc_std = \
                    df[(df['Subject'] == str(subject_number + 1)) & (df['Joint'] == joint_name) & (
                                df['Sensor'] == 'Parent Acc')][
                        'Std']
                child_acc_std = \
                    df[(df['Subject'] == str(subject_number + 1)) & (df['Joint'] == joint_name) & (
                                df['Sensor'] == 'Child Acc')][
                        'Std']
                min_acc_std = child_acc_std #np.min(parent_acc_std.values[0], child_acc_std.values[0])

                # Next, isolate the joint median errors from the subject error for each method
                method_errors = {method: np.mean(np.median(np.abs(np.array(method_data)) * 180 / np.pi, axis=0)) for method, method_data in
                                 subject_error[joint_name].items()}

                # Add data to scatter plot
                for method, errors in method_errors.items():
                    # for i, error in enumerate(errors):
                    plt.scatter(min_acc_std, errors, color=method_colors[method], label=method_display_names[method],
                                alpha=0.5, marker=axis_shape[subject_number])

        plt.xlabel("Minimum Accelerometer Standard Deviation")
        plt.ylabel("Median Joint Error")
        plt.legend(handles=[mpatches.Patch(color=color, label=label) for label, color in method_colors.items()])
        plt.show()

    generate_acc_std_vs_error_scatter_plot()