import os
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt  # <-- Import matplotlib
from src.toolchest.PlateTrial import PlateTrial
from src.RelativeFilterPlus import RelativeFilter


if __name__ == "__main__":
    
    # -----------------------------------------------------------------
    # 1. Load the plate trials
    # -----------------------------------------------------------------
    print("Step 1: Loading plate trials...")
    subject_num = '01'
    activity = 'complexTasks'
    subject_activity_folder = os.path.abspath(os.path.join("data", f"Subject{subject_num}", activity))
    
    try:
        plate_trials = PlateTrial.load_trial_from_folder(
            subject_activity_folder,
            align_plate_trials=True
        )
    except Exception as e:
        print(f"Failed to load data. Make sure your 'data/data/Subject01/walking' folder exists.")
        print(f"Error: {e}")
        exit()

    # Just pick one joint to keep it simple
    parent_name, child_name = 'pelvis_imu', 'femur_r_imu'
    
    parent_trial = next((p for p in plate_trials if p.name.__contains__(parent_name)), None)
    child_trial = next((p for p in plate_trials if p.name.__contains__(child_name)), None)

    if not parent_trial or not child_trial:
        print(f"Could not find trials for {parent_name} or {child_name}. Exiting.")
        exit()

    print(f"Successfully loaded {parent_trial.name} and {child_trial.name}.")
    num_frames = len(parent_trial)
    timestamps = parent_trial.imu_trace.timestamps[:num_frames]

    # -----------------------------------------------------------------
    # 2. Generate ground truth joint rotations
    # -----------------------------------------------------------------
    print("\nStep 2: Generating ground truth joint rotations...")
    
    # Get world rotations from the 'world_trace'
    R_wp_list = [Rotation.from_matrix(r) for r in parent_trial.world_trace.rotations]
    R_wc_list = [Rotation.from_matrix(r) for r in child_trial.world_trace.rotations]

    # R_pc = R_wp_inverse * R_wc
    R_pc_ground_truth_list = [(R_wp.inv() * R_wc) for R_wp, R_wc in zip(R_wp_list, R_wc_list)]

    # Convert the list of Rotation objects into a single stacked Rotation object
    # This is our ground truth (R_pc) and is also used to find the axis
    R_pc_ground_truth_stack = Rotation.from_matrix([rot.as_matrix() for rot in R_pc_ground_truth_list])
    
    # Estimate the axis from the ground truth data
    joint_axis_estimates, joint_axis_eigenvalues = parent_trial.world_trace.get_primary_joint_axis(child_trial.world_trace)
    print(f"Estimated joint axis (in parent frame): {joint_axis_estimates}")
    print(f"Estimated joint axis eigenvalues:  {joint_axis_eigenvalues}")

    joint_axis_estimate = joint_axis_estimates[0]  # Use the primary estimated axis

    # -----------------------------------------------------------------
    # 3. Use RelativeFilter to generate IMU-based rotations
    # -----------------------------------------------------------------
    print("\nStep 3: Generating IMU-based joint rotations using RelativeFilter...")

    # Create the filter
    joint_filter_with_joint_axis = RelativeFilter(gyro_std_parent=np.ones(3) * 0.01, gyro_std_child=np.ones(3) * 0.01,
                                  sensor_stds_parent=[np.ones(3) * 0.05, np.ones(3) * 0.05, np.ones(3) * 0.2], # [acc_std, mag_std, axis_std]
                                  sensor_stds_child=[np.ones(3) * 0.05, np.ones(3) * 0.05, np.ones(3) * 0.2]) # [acc_std, mag_std, axis_std]
    
    joint_filter_without_joint_axis = RelativeFilter(gyro_std_parent=np.ones(3) * 0.01, gyro_std_child=np.ones(3) * 0.01,
                                  sensor_stds_parent=[np.ones(3) * 0.05, np.ones(3) * 0.05], # [acc_std, mag_std]
                                  sensor_stds_child=[np.ones(3) * 0.05, np.ones(3) * 0.05]) # [acc_std, mag_std]

    # Initialize the filter with the *first* ground truth pose
    joint_filter_with_joint_axis.set_qs(R_wp_list[0], R_wc_list[0])
    joint_filter_without_joint_axis.set_qs(R_wp_list[0], R_wc_list[0])

    dt = np.mean(np.diff(timestamps))
    R_pc_with_joint_axis = []
    R_pc_without_joint_axis = []

    for t in range(num_frames):
        # Get raw sensor data from the 'imu_trace'
        parent_gyro = parent_trial.imu_trace.gyro[t]
        parent_acc = parent_trial.imu_trace.acc[t]
        parent_mag = parent_trial.imu_trace.mag[t]
        
        child_gyro = child_trial.imu_trace.gyro[t]
        child_acc = child_trial.imu_trace.acc[t]
        child_mag = child_trial.imu_trace.mag[t]

        # Pack the sensor readings
        sensor_readings_parent = [parent_acc, parent_mag]
        sensor_readings_child = [child_acc, child_mag]

        # Update the filter
        joint_filter_without_joint_axis.update(parent_gyro, child_gyro,
                            sensor_readings_parent,
                            sensor_readings_child, 
                            dt)
        
        # Add the joint axis as an additional "sensor" reading
        joint_filter_with_joint_axis.update(parent_gyro, child_gyro,
                            sensor_readings_parent + [joint_axis_estimate],
                            sensor_readings_child + [joint_axis_estimate],
                            dt)

        # Store the filter's estimate of the relative rotation
        R_pc_with_joint_axis.append(joint_filter_with_joint_axis.get_R_pc())
        R_pc_without_joint_axis.append(joint_filter_without_joint_axis.get_R_pc())

    print(f"Calculated {len(R_pc_with_joint_axis)} IMU-based relative rotations.")

    # -----------------------------------------------------------------
    # 4. Calculate and Print Error
    # -----------------------------------------------------------------
    print("\nStep 4: Calculating error...")

    # Stack the lists of estimates into single Rotation objects for vectorized calculation
    R_pc_est_with_axis = Rotation.from_matrix(R_pc_with_joint_axis)
    R_pc_est_without_axis = Rotation.from_matrix(R_pc_without_joint_axis)

    # Calculate the error rotation: R_err = R_true * R_est.inv()
    R_err_with_axis = R_pc_ground_truth_stack * R_pc_est_with_axis.inv()
    R_err_without_axis = R_pc_ground_truth_stack * R_pc_est_without_axis.inv()

    # Get the angle of the error rotation (geodesic distance) in degrees
    errors_with_axis_deg = np.degrees(R_err_with_axis.magnitude())
    errors_without_axis_deg = np.degrees(R_err_without_axis.magnitude())

    # Print the mean error
    print(f"Mean Error (Without Axis): {np.mean(errors_without_axis_deg):.4f} degrees")
    print(f"Mean Error (With Axis):    {np.mean(errors_with_axis_deg):.4f} degrees")


    # -----------------------------------------------------------------
    # 5. Plot Error
    # -----------------------------------------------------------------
    print("\nStep 5: Plotting results...")

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, errors_without_axis_deg, label='Error (Without Joint Axis)', alpha=0.8)
    plt.plot(timestamps, errors_with_axis_deg, label='Error (With Joint Axis)', linestyle='--', alpha=0.8)
    plt.title(f'IMU vs. Ground Truth Rotation Error ({parent_name} to {child_name})')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Error (degrees)')
    plt.legend()
    plt.grid(True)
    plt.show()


    # -----------------------------------------------------------------
    # 6. Print hello world
    # -----------------------------------------------------------------
    print("\nStep 6:")
    print("hello world")