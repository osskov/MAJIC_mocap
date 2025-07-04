import os.path
from typing import List, Dict

import numpy as np
from scipy.stats import wilcoxon, ttest_rel
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
    'axes.labelsize': 22,  # Larger axis label font size
    'xtick.labelsize': 22,  # Larger x-axis tick font size
    'ytick.labelsize': 22,  # Larger y-axis tick font size
    'legend.fontsize': 16,  # Larger legend font size
    'axes.labelweight': 'bold',  # Make axis labels bold
    'axes.titleweight': 'bold',  # Make titles bold
})

method_display_names = {
    'Mag Free': 'Magnetometer Free',
    'Never Project': 'MAJIC 0th Order',
    'Always Project': 'MAJIC 1st Order',
    'Cascade': 'MAJIC Adaptive'
}

method_colors = {
    method: color for method, color in zip(list(method_display_names.keys()), plt.cm.tab10.colors)
}

def add_significance_bar(ax, x1, x2, y, p_value, bar_height=1):
    """Add a bar with asterisks indicating significance level."""
    stars = ''
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    
    if stars:
        bar_y = y + bar_height
        ax.plot([x1, x1, x2, x2], [y, bar_y, bar_y, y], 'k-', linewidth=1)
        ax.text((x1 + x2) / 2, bar_y, stars, ha='center', va='bottom')

def plot_bars(bar_data, positions, labels, width, spacing, title, y_lim=[0, 40], keep_legend=True, p_values=None):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(4, 8))

    methods = [stats['label'] for stats in bar_data]
    methods = list(dict.fromkeys(methods))  # Remove duplicates

    # Plot the bars
    for idx, stats in enumerate(bar_data):
        method = stats['label']
        ax.bar(positions[idx], stats['mean'], width / len(methods) * 0.6,
               yerr=[[stats['mean'] - stats['q1']], [stats['q3'] - stats['mean']]],
               color=method_colors[method],
               capsize=5,
               label=method_display_names[method] if method not in [b.get_label() for b in ax.containers] else "")

    # Add significance bars if p_values are provided
    if p_values is not None:
        max_height = max([stats['q3'] for stats in bar_data])
        spacing = (y_lim[1] - y_lim[0]) * 0.05
        
        for i, p_dict in enumerate(p_values):
            if 'position' in p_dict:
                # For grouped comparisons (e.g., per joint)
                pos = p_dict['position']
                p_val = p_dict['p_value']
                x1 = pos - width/4
                x2 = pos + width/4
                add_significance_bar(ax, x1, x2, max_height + spacing, p_val, spacing)
            else:
                # For method comparisons
                method1, method2 = p_dict['methods']
                p_val = p_dict['p_value']
                idx1 = methods.index(method1)
                idx2 = methods.index(method2)
                x1 = positions[idx1]
                x2 = positions[idx2]
                add_significance_bar(ax, x1, x2, max_height + spacing, p_val, spacing)

    # Set x-ticks and labels
    group_positions = []
    current_pos = positions[0] + width / 4
    for _ in labels:
        group_positions.append(current_pos)
        current_pos += width + spacing

    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels, rotation=0)

    if keep_legend:
        ax.legend(title='Filter', loc='upper left')

    ax.set_title(title, pad=20)
    ax.set_ylabel('Error (degrees)')
    ax.set_ylim(y_lim)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

# Load the precomputed joint angle errors for each subject
subjects_errors: List[Dict] = []

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
                     joint_name, joint_data in subject_error.items() if '_2' not in joint_name}

    subjects_errors.append(subject_error)

def run_statistical_tests(data1, data2, method1_name, method2_name):
    """Run both Wilcoxon and paired t-test on the data."""
    # Wilcoxon signed-rank test
    w_stat, w_p = wilcoxon(data1, data2)
    
    # Paired t-test
    t_stat, t_p = ttest_rel(data1, data2)
    
    print(f"\nStatistical tests comparing {method1_name} vs {method2_name}:")
    print(f"Wilcoxon test - statistic: {w_stat:.4f}, p-value: {w_p:.4e}")
    print(f"Paired t-test - statistic: {t_stat:.4f}, p-value: {t_p:.4e}")
    
    return {
        'wilcoxon': {'statistic': w_stat, 'p_value': w_p},
        't_test': {'statistic': t_stat, 'p_value': t_p}
    }

def generate_all_joints_bar_plot():
    # Group errors by method
    method_errors = {method: np.array([]) for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']}
    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            for method, angle_errors in methods_data.items():
                angle_errors = np.array(angle_errors).flatten()
                method_errors[method] = np.concatenate([method_errors[method], angle_errors])

    print('Errors collected by method.')

    # Prepare the data for plotting
    bar_data = []
    positions = []
    labels = []
    current_pos = 1
    width = 1
    spacing = 0
    title = 'Distribution of All Joint Angle Errors for Each Sensor Fusion Filter'
    
    for method_name, method_error in method_errors.items():
        method_error = method_error * 180 / np.pi  # Convert to degrees
        plot_quantity = method_error

        # Calculate statistics
        stats = {
            'mean': np.mean(plot_quantity),
            'q1': np.percentile(plot_quantity, 30),
            'q3': np.percentile(plot_quantity, 70),
            'label': method_name
        }
        
        bar_data.append(stats)
        positions.append(current_pos)
        
        labels.append(
            method_display_names[method_name] + '\n(Mean Error: {:.2f} degrees)'.format(stats['mean']))
        current_pos += width + spacing

        print(f'Errors calculated for {method_name}.')
        print(f'Mean: {stats["mean"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
        print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')

        if method_name != 'Mag Free':
            _, p_value = wilcoxon(method_error, method_errors['Mag Free'])
            print(f'Wilcoxon signed-rank test p-value for {method_name} vs. Mag Free: {p_value}')
            if p_value < 0.05:
                print(f'{method_name} is significantly different from Mag Free.')
            else:
                print(f'{method_name} is not significantly different from Mag Free.')

    p_values = []
    for method_name in ['Never Project', 'Always Project', 'Cascade']:
        data1 = method_errors[method_name] * 180 / np.pi
        data2 = method_errors['Mag Free'] * 180 / np.pi
        
        test_results = run_statistical_tests(data1, data2, method_name, 'Mag Free')
        p_values.append({
            'methods': ['Mag Free', method_name],
            'p_value': test_results['wilcoxon']['p_value']
        })

    plot_bars(bar_data, positions, labels, width, spacing, title, y_lim=[-20, 20], keep_legend=False, p_values=p_values)

def generate_abs_val_all_joints_bar_plot():
    method_errors = {method: np.array([]) for method in ['Mag Free', 'Never Project', 'Always Project']}
    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                if method == 'Cascade':
                    continue
                angle_errors = np.array(np.abs(angle_errors)).flatten()
                method_errors[method] = np.concatenate([method_errors[method], angle_errors])

    print('Errors collected by method.')

    bar_data = []
    positions = []
    labels = []
    current_pos = 0.5
    width = 0.8
    spacing = 0.4
    title = 'Angle Error Distribution \nPer Sensor Fusion Filter'

    for method_name, method_error in method_errors.items():
        method_error = method_error * 180 / np.pi
        plot_quantity = np.abs(method_error)

        stats = {
            'mean': np.mean(plot_quantity),
            'q1': np.percentile(plot_quantity, 30),
            'q3': np.percentile(plot_quantity, 70),
            'label': method_name
        }
        
        bar_data.append(stats)
        positions.append(current_pos)
        labels.append(method_display_names[method_name])
        current_pos += width + spacing

        print(f'Errors calculated for {method_name}.')
        print(f'Mean: {stats["mean"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
        print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')

    p_values = []
    for method_name in ['Never Project', 'Always Project']:
        data1 = np.abs(method_errors[method_name] * 180 / np.pi)
        data2 = np.abs(method_errors['Mag Free'] * 180 / np.pi)
        
        test_results = run_statistical_tests(data1, data2, method_name, 'Mag Free')
        p_values.append({
            'methods': ['Mag Free', method_name],
            'p_value': test_results['wilcoxon']['p_value']
        })

    plot_bars(bar_data, positions, labels, width, spacing, title, y_lim=[0, 22], keep_legend=True, p_values=p_values)

def generate_per_axis_bar_plot():
    axis_errors = [{method: [] for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']} for _ in range(3)]
    axis_display_names = ['Flexion', 'Adduction', 'Rotation']

    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                for axis_idx in range(3):
                    axis_error = [error[axis_idx] for error in angle_errors]
                    axis_errors[axis_idx][method].extend(axis_error)

    print('Errors collected by axis.')

    bar_data = []
    positions = []
    width = 1
    spacing = 0.1
    title = 'Joint Angle Error Distribution per Axis'
    current_pos = 0.2

    p_values = []
    for i, axis_error_dict in enumerate(axis_errors):
        data1 = np.abs(np.array(axis_error_dict['Never Project']) * 180 / np.pi)
        data2 = np.abs(np.array(axis_error_dict['Mag Free']) * 180 / np.pi)
        
        test_results = run_statistical_tests(data1, data2, 'Never Project', 'Mag Free')
        p_values.append({
            'position': current_pos,
            'p_value': test_results['wilcoxon']['p_value']
        })
        print(f"\nAxis: {axis_display_names[i]}")
        current_pos += width + spacing

    for i, axis_error_dict in enumerate(axis_errors):
        num_axis_in_method = 3
        offsets = np.linspace(-width / 2, width / 2, num=num_axis_in_method + 2)[1:-1]
        
        for offset, method in zip(offsets, ['Mag Free', 'Never Project']):
            plot_quantity = np.abs(np.array(axis_error_dict[method])) * 180 / np.pi

            stats = {
                'mean': np.mean(plot_quantity),
                'q1': np.percentile(plot_quantity, 30),
                'q3': np.percentile(plot_quantity, 70),
                'label': method
            }
            
            bar_data.append(stats)
            positions.append(current_pos + offset)

            print(f'Errors calculated for {axis_display_names[i]} {method}.')
            print(f'Mean: {stats["mean"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
            print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')
            
        current_pos += width + spacing

    plot_bars(bar_data, positions, axis_display_names, width, spacing, title, p_values=p_values)

def generate_per_joint_bar_plot(title, joint_filter=None, method_filter=['Mag Free', 'Never Project', 'Always Project', 'Cascade']):
    joint_errors = {}

    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_filter and joint_name not in joint_filter:
                continue
            for method, angle_errors in methods_data.items():
                if method_filter and method not in method_filter:
                    continue
                for axis_idx in range(3):
                    if joint_name not in joint_errors.keys():
                        joint_errors[joint_name] = {method: [] for method in method_filter}
                    axis_error = [error[axis_idx] for error in angle_errors]
                    joint_errors[joint_name][method].extend(axis_error)

    print('Errors collected by joint.')

    bar_data = []
    positions = []
    labels = []
    current_pos = 1
    width = 1
    spacing = 0
    y_lim_max = 0
    last_dof_name = ''

    p_values = []
    for dof_name in joint_filter:
        if dof_name == '':  # Skip empty labels used for spacing
            continue
            
        dof_error_dict = joint_errors[dof_name]
        base_method = method_filter[0]  # Usually 'Mag Free'
        
        for method in method_filter[1:]:  # Compare other methods to base_method
            data1 = np.abs(np.array(dof_error_dict[method]) * 180 / np.pi)
            data2 = np.abs(np.array(dof_error_dict[base_method]) * 180 / np.pi)
            
            test_results = run_statistical_tests(data1, data2, method, base_method)
            p_values.append({
                'position': current_pos,
                'p_value': test_results['wilcoxon']['p_value']
            })
            print(f"\nJoint: {dof_name}")
        
        current_pos += width + spacing

    for dof_name in joint_filter:
        dof_error_dict = joint_errors[dof_name]

        if last_dof_name and (last_dof_name.split(' ')[0] != dof_name.split(' ')[0] or
                              ('Lumbar' in dof_name and last_dof_name.split(' ')[1] != dof_name.split(' ')[1]) or
                              ('Shoulder' in dof_name and last_dof_name.split(' ')[1] != dof_name.split(' ')[1])):
            current_pos += 1
            labels.append('')

        num_methods_in_dof = len(method_filter)
        offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_dof + 2)[1:-1]

        for offset, method in zip(offsets, method_filter):
            plot_quantity = np.abs(np.array(dof_error_dict[method])) * 180 / np.pi

            stats = {
                'mean': np.mean(plot_quantity),
                'q1': np.percentile(plot_quantity, 30),
                'q3': np.percentile(plot_quantity, 70),
                'label': method
            }
            
            if stats['q3'] > y_lim_max:
                y_lim_max = stats['q3'] + 5
                
            bar_data.append(stats)
            positions.append(current_pos + offset)

            print(f'Errors calculated for {method}.')
            print(f'Mean: {stats["mean"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
            print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')
            
        labels.append(dof_name)
        current_pos += width + spacing
        last_dof_name = dof_name

    plot_bars(bar_data, positions, labels, width, spacing, title, y_lim=[0, y_lim_max], p_values=p_values)

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

    for subject_number, subject_error in enumerate(subjects_errors):
        if subject_number != 0:
            continue
        for joint_name, methods_data in subject_error.items():
            if joint_name != 'Hip':
                continue
            for dof, dof_name in enumerate(['Flexion', 'Adduction', 'Rotation']):
                timestamps = [time * 0.005 for time in range(len(np.array(methods_data['Mag Free'])[:, 0]))]
                subject_activity_timestamps = activity_timestamps[subject_number]
                legend_patches = []
                for index in range(len(subject_activity_timestamps) - 1):
                    start = subject_activity_timestamps[index]
                    end = subject_activity_timestamps[index + 1]
                    activity = activity_order[index % 5]

                    if activity == 'Walking':
                        color = 'black'
                        alpha = 0.25
                    elif activity == 'Running':
                        color = 'black'
                        alpha = 0.4
                    elif activity == 'Stairs and Side Stepping':
                        color = 'black'
                        alpha = 0.15
                    elif activity == 'Standing and Sitting':
                        color = 'green'
                        alpha = 0.05

                    if end <= 5000:
                        continue
                    if start < 5000:
                        start = 5000
                    plt.axvspan((start - 5000) * 0.005, (end - 5000) * 0.005, color=color, alpha=alpha, linewidth=0)
                    if activity not in [patch.get_label() for patch in legend_patches]:
                        legend_patches.append(mpatches.Patch(color=color, alpha=alpha, label=activity))

                for method in methods_data.keys():
                    plt.plot(timestamps, np.array(methods_data[method])[:, dof] * 180.0 / np.pi,
                             label=method)
                    plt.title(f"Subject {subject_number + 1} {joint_name} {dof_name} Joint Error over Time per Filter")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Angle Error (degrees)")

                plt.legend(handles=legend_patches + plt.gca().get_legend_handles_labels()[0], loc='upper left', bbox_to_anchor=(1, 1))
                plt.show()

if True:
    print('Generating Figure 4')
    generate_per_joint_bar_plot('Joint Angle Error Distribution per Joint', 
                              joint_filter=['Ankle', 'Knee', 'Hip', 'Lumbar_1', 'Shoulder_1', 'Elbow'], 
                              method_filter=['Mag Free', 'Never Project'])

print('Done') 