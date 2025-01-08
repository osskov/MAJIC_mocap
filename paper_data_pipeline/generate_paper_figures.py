import os.path
from typing import List, Dict

import numpy as np
from scipy.stats import wilcoxon
from src.IMUSkeleton import IMUSkeleton, RawReading
from src.toolchest.PlateTrial import PlateTrial
from paper_data_pipeline.generate_kinematics_from_precomputed_plates import generate_kinematics_report
import nimblephysics as nimble
import matplotlib.pyplot as plt

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

    methods = list(set(stats['label'] for stats in boxplot_data))

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
subject_errors: List[Dict] = []
subject_projected_readings: List[Dict[str, Dict[str, List[np.array]]]] = [] # List by subject, of dicts with joint names as keys and sensor data lists as values

load_kinematics = True
load_sensor_data = True
# Collect kinematics
for subject_number in range(1,2):
    if load_kinematics:
        subject_error = generate_kinematics_report(
            b3d_path=f"../data/S{subject_number}_IMU_Data/S{subject_number}.b3d",
            model_path=f"../data/S{subject_number}_IMU_Data/SimplifiedIMUModel.osim",
            cached_history_path=f"../data/S{subject_number}_IMU_Data/Walking/cached_skeleton_history.npz",
            output_path=f"../data/S{subject_number}_IMU_Data/Walking/",
            output_csv=f"../data/Walking_Results/S{subject_number}_Walking_joint_angles.csv"
        )
        subject_errors.append(subject_error)

    if load_sensor_data:
        subject_skeleton = IMUSkeleton.load_from_json(f"../data/S{subject_number}_IMU_Data/Walking/skeleton.json")
        plate_trials = PlateTrial.load_cheeseburger_trial_from_folder(f"../data/S{subject_number}_IMU_Data/Walking/")

        all_readings_list: List[Dict[str, RawReading]] = RawReading.create_readings_from_plate_trials(plate_trials)
        all_readings_list = all_readings_list[5000:48000]

        projected_reading_list = {joint_name: {'parent_acc': [], 'parent_avg_mag': [], 'parent_proj_mag': [],
                                               'child_acc': [], 'child_avg_mag': [], 'child_proj_mag': []} for joint_name in
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
                    readings_dict[child_name] for child_name in readings_dict if child_name.__contains__(joint.child_name))

                parent_reading, child_reading = joint.joint_filter.readings_projector.get_projected_readings_at_joint_center(
                    parent_reading=raw_parent_reading, child_reading=raw_child_reading)

                projected_reading_list[joint_name]['parent_acc'].append(parent_reading.acc)
                projected_reading_list[joint_name]['parent_avg_mag'].append(np.mean(np.array(raw_parent_reading.mags), axis=0))
                projected_reading_list[joint_name]['parent_proj_mag'].append(parent_reading.mag)
                projected_reading_list[joint_name]['child_acc'].append(child_reading.acc)
                projected_reading_list[joint_name]['child_avg_mag'].append(np.mean(np.array(raw_child_reading.mags), axis=0))
                projected_reading_list[joint_name]['child_proj_mag'].append(child_reading.mag)

        subject_projected_readings.append(projected_reading_list)

def generate_all_joints_box_plot():
    # From just this data, we can create our first figure: a boxplot of the errors for each method
    # First we need to group the errors by method
    method_errors = {method: np.array([]) for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']}
    for subject_kinematics in subject_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                angle_errors = np.array(angle_errors).flatten()
                method_errors[method] = np.concatenate([method_errors[method], angle_errors])

    print('Errors collected by method.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 0.5  # Total width allocated for each group of boxplots
    spacing = 0  # Spacing between groups
    title = 'Distribution of All Joint Angle Errors for Each Sensor Fusion Filter'
    for method_name, method_error in method_errors.items():
        method_error = method_error * 180 / np.pi  # Convert to degrees

        plot_quantity = method_error

        # Remove outliers based on 1.5 * IQR rule
        q1 = np.percentile(plot_quantity, 25)
        q3 = np.percentile(plot_quantity, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]
        outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]

        # Print number of data points considered outliers
        print(f'Number of outliers for {method_name}: {len(method_error) - len(plot_quantity)}')

        stats = {
            'med': np.median(plot_quantity),
            'q1': np.percentile(plot_quantity, 30),
            'q3': np.percentile(plot_quantity, 70),
            'whislo': np.min([lower_bound, np.min(plot_quantity)]),  # Ensure that the whiskers don't go below 0
            'whishi': np.max([upper_bound, np.max(plot_quantity)]),
            'fliers': outliers,
            'label': method_name
        }
        boxplot_data.append(stats)
        # Calculate position for this boxplot
        pos = current_pos
        positions.append(pos)

        labels.append(
            method_display_names[method_name] + '\n(Median Error: {:.2f} degrees)'.format(np.median(plot_quantity)))
        current_pos += width + spacing  # Move to the next group position

        print(f'Errors calculated for {method_name}.')
        print(f'Median: {stats["med"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
        print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')

        # Perform Wilcoxon signed-rank test against the Mag Free method
        if method_name == 'Mag Free':
            continue
        _, p_value = wilcoxon(method_error, method_errors['Mag Free'])
        print(f'Wilcoxon signed-rank test p-value for {method_name} vs. Mag Free: {p_value}')
        if p_value < 0.05:
            print(f'{method_name} is significantly different from Mag Free.')
        else:
            print(f'{method_name} is not significantly different from Mag Free.')

    plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[-20, 20], keep_legend=False)


def generate_abs_val_all_joints_box_plot():
    # From just this data, we can create our first figure: a boxplot of the errors for each method
    # First we need to group the errors by method
    method_errors = {method: np.array([]) for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']}
    for subject_kinematics in subject_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                angle_errors = np.array(np.abs(angle_errors)).flatten()
                method_errors[method] = np.concatenate([method_errors[method], angle_errors])

    print('Errors collected by method.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 0.5  # Total width allocated for each group of boxplots
    spacing = 0  # Spacing between groups
    title = 'Distribution of All Joint Angle Error Magnitudes for Each Sensor Fusion Filter'
    for method_name, method_error in method_errors.items():
        method_error = method_error * 180 / np.pi  # Convert to degrees

        plot_quantity = np.abs(method_error)

        # Remove outliers based on 1.5 * IQR rule
        q1 = np.percentile(plot_quantity, 25)
        q3 = np.percentile(plot_quantity, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr if q1 - 1.5 * iqr > 0 else min(plot_quantity)
        upper_bound = q3 + 1.5 * iqr

        plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]
        outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]

        # Print number of data points considered outliers
        print(f'Number of outliers for {method_name}: {len(method_error) - len(plot_quantity)}')

        stats = {
            'med': np.median(plot_quantity),
            'q1': np.percentile(plot_quantity, 30),
            'q3': np.percentile(plot_quantity, 70),
            'whislo': np.max([lower_bound, np.min(plot_quantity)]),  # Ensure that the whiskers don't go below 0
            'whishi': np.max([upper_bound, np.max(plot_quantity)]),
            'fliers': outliers,
            'label': method_name
        }

        boxplot_data.append(stats)
        # Calculate position for this boxplot
        pos = current_pos
        positions.append(pos)

        labels.append(
            method_display_names[method_name] + '\n(Median Error: {:.2f} degrees)'.format(np.median(method_error)))
        current_pos += width + spacing  # Move to the next group position

        print(f'Errors calculated for {method_name}.')
        print(f'Median: {stats["med"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
        print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')

        # Perform Wilcoxon signed-rank test against the Mag Free method
        if method_name == 'Mag Free':
            continue
        _, p_value = wilcoxon(method_error, method_errors['Mag Free'])
        print(f'Wilcoxon signed-rank test p-value for {method_name} vs. Mag Free: {p_value}')
        if p_value < 0.05:
            print(f'{method_name} is significantly different from Mag Free.')
        else:
            print(f'{method_name} is not significantly different from Mag Free.')

    plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[0, 30], keep_legend=False)


def generate_per_axis_box_plot():
    # First we need to split our data by axis (flexion, abduction, and rotation)
    axis_errors = [{method: [] for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']} for _ in
                   range(3)]
    axis_display_names = ['Flexion', 'Abduction', 'Rotation']

    for subject_kinematics in subject_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                for axis_idx in range(3):
                    axis_error = [error[axis_idx] for error in angle_errors]
                    axis_errors[axis_idx][method].extend(axis_error)

    print('Errors collected by axis.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 1  # Total width allocated for each group of boxplots (per joint)
    spacing = 0.5  # Spacing between groups
    title = 'Joint Angle Error Distribution per Axis'
    for i, axis_error_dict in enumerate(axis_errors):
        method_data_list = []

        # Get the methods present for this joint
        methods_in_activity = list(axis_error_dict.keys())
        num_methods_in_activity = len(methods_in_activity)
        # Calculate offsets for methods within the joint group
        offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_activity + 2)[1:-1]

        for offset, method in zip(offsets, methods_in_activity):
            if 'Markers' in method:
                continue

            plot_quantity = np.array(axis_error_dict[method]) * 180 / np.pi  # Convert to degrees

            # Remove outliers based on 1.5 * IQR rule
            q1 = np.percentile(plot_quantity, 25)
            q3 = np.percentile(plot_quantity, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]
            outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]

            # Print number of data points considered outliers
            print(
                f'Number of outliers for {method} in {axis_display_names[i]}: {len(axis_error_dict[method]) - len(plot_quantity)}')

            # Extract statistics required for the boxplot
            stats = {
                'med': np.median(plot_quantity),
                'q1': q1,
                'q3': q3,
                'whislo': np.min([lower_bound, np.min(plot_quantity)]),
                'whishi': np.max([upper_bound, np.max(plot_quantity)]),
                'mean': np.mean(plot_quantity),
                'fliers': outliers,
                'label': method
            }
            boxplot_data.append(stats)
            # Calculate position for this boxplot
            pos = current_pos + offset
            positions.append(pos)
            method_data_list.append(stats)

            print(f'Errors calculated for {method}.')
            print(f'Median: {stats["med"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
            print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')

        labels.append(axis_display_names[i] + '\n(Median Error: {:.2f} degrees)'.format(np.median(plot_quantity)))
        current_pos += width + spacing  # Move to the next group position

    plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[-70, 70])


def generate_abs_val_per_axis_box_plot():
    # First we need to split our data by axis (flexion, abduction, and rotation)
    axis_errors = [{method: [] for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']} for _ in
                   range(3)]
    axis_display_names = ['Flexion', 'Abduction', 'Rotation']

    for subject_kinematics in subject_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                for axis_idx in range(3):
                    axis_error = [error[axis_idx] for error in angle_errors]
                    axis_errors[axis_idx][method].extend(axis_error)

    print('Errors collected by axis.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 1  # Total width allocated for each group of boxplots (per joint)
    spacing = 0.5  # Spacing between groups
    title = 'Joint Angle Error Magnitude Distribution per Axis'
    for i, axis_error_dict in enumerate(axis_errors):
        method_data_list = []

        # Get the methods present for this joint
        methods_in_activity = list(axis_error_dict.keys())
        num_methods_in_activity = len(methods_in_activity)
        # Calculate offsets for methods within the joint group
        offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_activity + 2)[1:-1]

        for offset, method in zip(offsets, methods_in_activity):
            if 'Markers' in method:
                continue

            plot_quantity = np.abs(np.array(axis_error_dict[method])) * 180 / np.pi  # Convert to degrees

            # Print number of data points considered outliers
            print(
                f'Number of outliers for {method} in {axis_display_names[i]}: {len(axis_error_dict[method]) - len(plot_quantity)}')

            # Remove outliers based on 1.5 * IQR rule
            q1 = np.percentile(plot_quantity, 25)
            q3 = np.percentile(plot_quantity, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr if q1 - 1.5 * iqr > 0 else min(plot_quantity)
            upper_bound = q3 + 1.5 * iqr
            outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]
            plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]

            # Extract statistics required for the boxplot
            stats = {
                'med': np.median(plot_quantity),
                'q1': q1,
                'q3': q3,
                'whislo': np.max([lower_bound, np.min(plot_quantity)]),
                'whishi': np.max([upper_bound, np.max(plot_quantity)]),
                'mean': np.mean(plot_quantity),
                'fliers': outliers,
                'label': method
            }
            boxplot_data.append(stats)
            # Calculate position for this boxplot
            pos = current_pos + offset
            positions.append(pos)
            method_data_list.append(stats)

        labels.append(axis_display_names[i] + '\n(Median Error: {:.2f} degrees)'.format(np.median(plot_quantity)))
        current_pos += width + spacing  # Move to the next group position

    plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[0, 70])


def generate_per_subject_per_method_scatter_plot():
    # Select 30 random points per joint per subject per method
    num_points = 30
    subjects_data_points = [
        {method: [] for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade', 'Markers']} for _ in
        range(3)]
    for subject_kinematics in subject_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                for axis_idx in range(3):
                    axis_error = [error[axis_idx] for error in angle_errors]
                    if method == 'Markers':
                        subjects_data_points[axis_idx][method].extend(axis_error)
                    else:
                        subjects_data_points[axis_idx][method].extend(np.random.choice(axis_error, num_points))

def generate_mag_norm_diff_per_projection_error_plot():
    # Generate a scatter plot of parent and child magnetic field estimate magnitude difference. Scatterplot against the MAJIC 0 and MAJIC 1 joint errors for ankle and lumbar.
    # First, pull out random 100 samples
    random_indeces = np.random.choice(len(subject_projected_readings), 100)

    # First, isolate ankle and elbow joint errors for each subject
    ankle_errors = {method: [] for method in ['Never Project', 'Always Project']}
    elbow_errors = {method: [] for method in ['Never Project', 'Always Project']}

    for subject_kinematics in subject_errors:
        for joint_name, methods_data in subject_kinematics.items():
            for method, angle_errors in methods_data.items():
                angle_errors = np.array(angle_errors)[random_indeces]
                if joint_name == 'ankle':
                    ankle_errors[method].extend(angle_errors)
                elif joint_name == 'elbow':
                    elbow_errors[method].extend(angle_errors)

    # Now isolate the unprojected and projected magnetic field magnitude differences for the ankle and elbow
    ankle_parent_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['ankle']['parent_avg_mag'])) for i in random_indeces]
    ankle_child_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['ankle']['child_avg_mag'])) for i in random_indeces]
    ankle_mag_diffs = np.abs(np.array(ankle_parent_norms) - np.array(ankle_child_norms))

    ankle_projected_parent_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['ankle']['parent_proj_mag'])) for i in random_indeces]
    ankle_projected_child_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['ankle']['child_proj_mag'])) for i in random_indeces]
    ankle_projected_mag_diffs = np.abs(np.array(ankle_projected_parent_norms) - np.array(ankle_projected_child_norms))

    elbow_parent_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['elbow']['parent_avg_mag'])) for i in random_indeces]
    elbow_child_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['elbow']['child_avg_mag'])) for i in random_indeces]
    elbow_mag_diffs = np.abs(np.array(elbow_parent_norms) - np.array(elbow_child_norms))

    elbow_projected_parent_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['elbow']['parent_proj_mag'])) for i in random_indeces]
    elbow_projected_child_norms = [np.linalg.norm(np.array(subject_projected_readings[i]['elbow']['child_proj_mag'])) for i in random_indeces]
    elbow_projected_mag_diffs = np.abs(np.array(elbow_projected_parent_norms) - np.array(elbow_projected_child_norms))

    # Now plot the scatter plot
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(ankle_mag_diffs, ankle_errors['Never Project'], label='Ankle - Unprojected', color='orange', shape='s')
    ax.scatter(ankle_projected_mag_diffs, ankle_errors['Always Project'], label='Ankle - Projected', color='green', shape='s')
    ax.scatter(elbow_mag_diffs, elbow_errors['Never Project'], label='Elbow - Unprojected', color='orange')
    ax.scatter(elbow_projected_mag_diffs, elbow_errors['Always Project'], label='Elbow - Projected', color='green')

    ax.set_xlabel('Parent and Child Magnetic Field Magnitude Difference')
    ax.set_ylabel('Joint Angle Error (degrees)')
    ax.set_title('Parent and Child Magnetic Field Magnitude Difference vs. Joint Angle Error')
    ax.legend()
    plt.tight_layout()


def generate_joint_error_vs_time_plot_activity_split_plot():
    activity_order = ['Walking', 'Running', 'Stairs and Side Stepping', 'Standing and Sitting',
                      'Stairs and Side Stepping']
    activity_timestamps = [
        [0, 2300, 3400, 5000, 8100, 10100, 11900, 13000, 14400, 18000, 20200, 22400, 23600, 25000, 28800, 30700,
         32800, 34000, 35200, 38500, 41000, 43500, 44900, 46300, 48000],
        [0, 2400, 3500, 4600, 7000, 9300, 10800, 11800, 14000, 15300, 17000, 18700, 19800, 21500, 22400, 24100,
         25500, 27600, 30200, 32100, 33800, 36000, 37600, 38900, 40800, 42900, 44800, 46000, 47400, 48000],
        [0, 2400, 4000, 5700, 8800, 12400, 13800, 15000, 16300, 18800, 21000, 22700, 23700, 25500, 30000, 32000,
         33700, 35100, 36500, 40700, 43600, 45800, 46900, 48000]]

    for subject_error in subject_errors:
        for joint_name, methods_data in subject_error.items():
            timestamps = [time * 0.005 for time in range(len(np.array(methods_data['Mag Free'])[:, 0]))]
            activity_timestamps = activity_timestamps[1]

            for index in range(len(activity_timestamps) - 1):
                start = activity_timestamps[index]
                end = activity_timestamps[index + 1]
                activity = activity_order[index % 5]

                if activity == 'Walking':
                    color = 'black'
                    alpha = 0.1
                elif activity == 'Running':
                    color = 'red'
                    alpha = 0.1
                elif activity == 'Stairs and Side Stepping':
                    color = 'blue'
                    alpha = 0.1
                elif activity == 'Standing and Sitting':
                    color = 'green'
                    alpha = 0.1

                if end <= 5000:
                    continue
                if start < 5000:
                    start = 5000
                plt.axvspan((start - 5000) * 0.005, (end - 5000) * 0.005, color=color, alpha=alpha)

            for method in methods_data.keys():
                for dof, dof_name in enumerate(['Flexion', 'Adduction', 'Rotation']):
                    plt.plot(timestamps, np.array(methods_data[method])[:, dof] * 180.0 / np.pi,
                                     label=method + " " + dof_name)
                    plt.title(f"{joint_name} {dof} Joint Error over Time per Filter")
                    plt.xlabel("Time (seconds)")
                    plt.ylabel("Angle Error (degrees)")
                    plt.legend()
                    plt.show()

print('Generating figures...')

if load_kinematics and False:
    generate_per_axis_box_plot()
    generate_abs_val_per_axis_box_plot()
    generate_all_joints_box_plot()
    generate_abs_val_all_joints_box_plot()
    generate_per_subject_per_method_scatter_plot()


if load_kinematics and load_sensor_data:
    generate_mag_norm_diff_per_projection_error_plot()