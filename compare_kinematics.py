import numpy as np
from typing import Dict
from toolchest.KinematicsTrace import KinematicsTrace
import matplotlib.pyplot as plt
import pandas as pd

# Load statistics CSV
df = pd.read_csv('data/ODay_Data/all_trial_statistics.csv')

# Subject and activity setup
for subject in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']:
    subject = f"Subject{subject}"
    for activity in ['walking', 'complexTasks']:
        # File paths for different methods
        mot_files = {
            "ground truth": f"data/ODay_Data/{subject}/{activity}/Mocap/ikResults/{activity}_IK_trimmed.mot",
            "madgwick": f"data/ODay_Data/{subject}/{activity}/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/{activity}_IK.mot" if activity == 'walking' else f"data/ODay_Data/{subject}/{activity}/IMU/madgwick/IKResults/IKUniformWeights/{activity}_IK.mot",
            "markers": f"data/ODay_Data/{subject}/{activity}/Mocap/ikResults/IKWithErrorsUniformWeights/{activity}_IK.mot",
            "mag free": f"data/ODay_Data/{subject}/{activity}/IMU/Mag Free/IKResults/IKWithErrorsUniformWeights/{activity}_IK.mot",
            "unprojected": f"data/ODay_Data/{subject}/{activity}/IMU/Unprojected/IKResults/IKWithErrorsUniformWeights/{activity}_IK.mot",
            "never project": f"data/ODay_Data/{subject}/{activity}/IMU/Never Project/IKResults/IKWithErrorsUniformWeights/{activity}_IK.mot",
        }
        try:
            # Load all kinematics traces
            kinematics = {
                name: KinematicsTrace.load_kinematics_from_mot_file(mot_file)
                for name, mot_file in mot_files.items()
            }
            print(f"Loaded kinematics for {subject} {activity}.")
        except:
            print(f"Failed to load kinematics for {subject} {activity}. Skipping...")
            continue

        def plot_kinematics_errors(kinematics: Dict[str, KinematicsTrace], df: pd.DataFrame) -> pd.DataFrame:
            """
            Generate bar chart comparing kinematic RMSEs and write to DataFrame.
            """
            kinematics_differences = {
                name: kinematics['ground truth'][:60000] - kinematics[name][:60000]
                for name in kinematics if name != 'ground truth'
            }

            num_rows = 3
            num_cols = 5
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12), sharey=True)
            fig.suptitle("Kinematic Error Comparison", fontsize=16)
            ax_idx = 0

            ignore_keywords = ["time", "pelvis", "arm", "elbow", "pro", "beta"]

            for joint_name in kinematics['ground truth'].joint_angles.keys():
                if any(keyword in joint_name.lower() for keyword in ignore_keywords):
                    continue

                row_plotted = False
                for name, kinematics_trace in kinematics_differences.items():
                    if joint_name not in kinematics_trace.joint_angles:
                        print(f"Joint {joint_name} not found in {name}. Skipping...")
                        continue

                    rmse = np.mean(np.array(kinematics_trace.joint_angles[joint_name]) ** 2) ** 0.5
                    std = np.std(np.array(kinematics_trace.joint_angles[joint_name]))

                    mask = (
                            (df['subject'] == subject) &
                            (df['activity'] == activity) &
                            (df['joint'] == joint_name) &
                            (df['method'] == name)
                    )

                    if mask.any():
                        # Row already exists → update RMSE and STD only
                        df.loc[mask, ['rmse', 'std']] = rmse, std
                    else:
                        # Row doesn’t exist → create a new one
                        new_row = {
                            'subject': subject,
                            'activity': activity,
                            'joint': joint_name,
                            'method': name,
                            'rmse': rmse,
                            'std': std
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                    ax = axes[ax_idx // num_cols, ax_idx % num_cols]
                    ax.bar(name, rmse, yerr=std, label=name, alpha=0.7)
                    ax.set_title(joint_name)
                    ax.set_xlabel("Method")
                    ax.set_ylabel("Orientation Error (degrees)")
                    # ax.grid(True)
                    row_plotted = True

                if row_plotted:
                    ax.legend()
                    ax_idx += 1

            plt.tight_layout()
            plt.show()
            return df

        # Run plotting and update the CSV
        df = plot_kinematics_errors(kinematics, df)
df.to_csv('data/ODay_Data/all_trial_statistics.csv', index=False)