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


def plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[0, 40], keep_legend=True):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the boxplots
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')

    methods = [stats['label'] for stats in boxplot_data]
    methods = list(dict.fromkeys(methods))  # Remove duplicates

    # Since we have custom positions, we need to use bxp
    for idx, stats in enumerate(boxplot_data):
        bxp_stats = [stats]
        method = stats['label']
        ax.bxp(bxp_stats, positions=[positions[idx]], widths=width / len(methods) * 0.5,
               showfliers=False,
               boxprops=dict(facecolor=method_colors[method], color=method_colors[method]),
               medianprops=medianprops,
               whiskerprops=dict(color=method_colors[method]),
               capprops=dict(color=method_colors[method]),
               flierprops=dict(markeredgecolor=method_colors[method]),
               patch_artist=True  # This line enables 'facecolor' in boxprops
               )

    # Set x-ticks and labels
    group_positions = []
    current_pos = 1
    for _ in labels:
        group_positions.append(current_pos)
        current_pos += width + spacing

    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    if keep_legend:
        # Create a legend for the methods
        handles = [plt.Line2D([0], [0], color=method_colors[method], lw=10) for method in methods]
        ax.legend(handles, method_display_names, title='Filter', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(title)
    ax.set_ylabel('Error (degrees)')
    ax.set_ylim(y_lim)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# Load the precomputed joint angle errors for each subject
subjects_errors: List[Dict] = []
subjects_projected_readings: List[Dict[str, Dict[
    str, List[np.array]]]] = []  # List by subject, of dicts with joint names as keys and sensor data lists as values

load_kinematics = False
load_sensor_data = True
# Collect kinematics
for subject_number in range(1, 2):
    if load_kinematics:
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

    if load_sensor_data:
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


def generate_mag_norm_diff_per_projection_error_plot():
    # Generate a scatter plot of parent and child magnetic field estimate magnitude difference. Scatterplot against the MAJIC 0 and MAJIC 1 joint errors for ankle and lumbar.
    # First, pull out random 100 samples
    num_samples = 100
    random_indeces = np.random.choice(43000, num_samples)

    # First, isolate ankle and elbow joint errors for each subject
    ankle_errors = {method: [] for method in ['Never Project', 'Always Project']}
    elbow_errors = {method: [] for method in ['Never Project', 'Always Project']}

    # Now plot the scatter plot
    fig, ax = plt.subplots(figsize=(15, 8))

    for subject_number, subject_errors in enumerate(subjects_errors):
        for joint_name, methods_data in subject_errors.items():
            for method, angle_errors in methods_data.items():
                if method not in ['Never Project', 'Always Project']:
                    continue
                angle_errors = np.array(angle_errors)[random_indeces]
                angle_errors = angle_errors[:, 2] * 180 / np.pi  # Convert to degrees
                if joint_name == 'Ankle':
                    ankle_errors[method].extend(angle_errors)
                elif joint_name == 'Elbow':
                    elbow_errors[method].extend(angle_errors)

        # Now isolate the unprojected and projected magnetic field magnitude differences for the ankle and elbow
        ankle_parent_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Ankle']['parent_avg_mag'][i])) for i in
            random_indeces]
        ankle_child_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Ankle']['child_avg_mag'][i])) for i in
            random_indeces]
        ankle_mag_diffs = np.abs(np.array(ankle_parent_norms) - np.array(ankle_child_norms))

        ankle_projected_parent_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Ankle']['parent_proj_mag'][i])) for i
            in random_indeces]
        ankle_projected_child_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Ankle']['child_proj_mag'][i])) for i in
            random_indeces]
        ankle_projected_mag_diffs = np.abs(
            np.array(ankle_projected_parent_norms) - np.array(ankle_projected_child_norms))

        elbow_parent_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Elbow']['parent_avg_mag'][i])) for i in
            random_indeces]
        elbow_child_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Elbow']['child_avg_mag'][i])) for i in
            random_indeces]
        elbow_mag_diffs = np.abs(np.array(elbow_parent_norms) - np.array(elbow_child_norms))

        elbow_projected_parent_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Elbow']['parent_proj_mag'][i])) for i
            in random_indeces]
        elbow_projected_child_norms = [
            np.linalg.norm(np.array(subjects_projected_readings[subject_number]['Elbow']['child_proj_mag'][i])) for i in
            random_indeces]
        elbow_projected_mag_diffs = np.abs(
            np.array(elbow_projected_parent_norms) - np.array(elbow_projected_child_norms))

        for axis in range(3 * len(subjects_errors)):
            ax.scatter(ankle_mag_diffs, ankle_errors['Never Project'][axis * num_samples:(axis + 1) * num_samples],
                       label='Ankle - Unprojected', color='orange', marker='s')
            ax.scatter(ankle_projected_mag_diffs,
                       ankle_errors['Always Project'][axis * num_samples:(axis + 1) * num_samples],
                       label='Ankle - Projected', color='green', marker='s')
            ax.scatter(elbow_mag_diffs, elbow_errors['Never Project'][axis * num_samples:(axis + 1) * num_samples],
                       label='Elbow - Unprojected', color='orange')
            ax.scatter(elbow_projected_mag_diffs,
                       elbow_errors['Always Project'][axis * num_samples:(axis + 1) * num_samples],
                       label='Elbow - Projected', color='green')

            ax.set_xlabel('Parent and Child Magnetic Field Magnitude Difference')
            ax.set_ylabel('Joint Angle Error (degrees)')
            ax.set_title('Parent and Child Magnetic Field Magnitude Difference vs. Joint Angle Error')
            ax.legend()

    plt.tight_layout()
    plt.show()

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
