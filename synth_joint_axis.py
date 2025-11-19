import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.IMUTrace import IMUTrace

def calculate_axis_error_deg(v_true, v_est):
    """
    Calculates the angular error between two 3D vectors in degrees,
    accounting for the ambiguity in axis direction (v is the same as -v).
    """
    v1 = v_true / np.linalg.norm(v_true)
    v2 = v_est / np.linalg.norm(v_est)
    dot_product = np.abs(np.dot(v1, v2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.rad2deg(angle_rad)

def demonstrate_joint_center_estimation():
    """
    Creates a parent trial, a connected child trial, estimates the joint
    center between them, and compares the result to the ground truth.
    """
    print("Step 1: Generating a random 'parent' plate trial...")
    parent_plate_trial = PlateTrial.generate_random_plate_trial(
        duration=15, fs=100, add_noise=False
    )
    print("  - Parent trial created.")

    print("\nStep 2: Defining ground truth joint parameters...")
    dof = 3
    gt_offset_parent = np.random.rand(3)
    gt_offset_child = np.random.rand(3)
    
    print(f"  - Degrees of Freedom: {dof}")
    print(f"  - Ground Truth Parent Offset (parent frame): {gt_offset_parent}")
    print(f"  - Ground Truth Child Offset (child frame):   {gt_offset_child}")

    print("\nStep 3: Generating a connected 'child' plate trial...")
    # This call works as-is, as the new generator will create random internal axes
    # when they are not provided, which is sufficient for this test.
    child_plate_trial = parent_plate_trial.generate_random_connected_plate_trial(
        dof=dof,
        joint_center_parent=gt_offset_parent,
        joint_center_child=gt_offset_child,
        add_noise=False
    )
    print("  - Child trial generated successfully.")

    print("\nStep 4: Estimating joint center using `WorldTrace.get_joint_center()`...")
    est_offset_parent, est_offset_child, residual_error = parent_plate_trial.world_trace.get_joint_center(
        child_plate_trial.world_trace
    )
    print("  - Estimation complete.")

    print("\n" + "="*25)
    print("      RESULTS COMPARISON")
    print("="*25)
    
    parent_error_norm = np.linalg.norm(gt_offset_parent - est_offset_parent)
    print(f"\nParent Offset (Ground Truth): {np.round(gt_offset_parent, 5)}")
    print(f"Parent Offset (Estimated):    {np.round(est_offset_parent, 5)}")
    print(f"--> Parent Estimation Error (L2 Norm): {parent_error_norm:.6f} meters")

    child_error_norm = np.linalg.norm(gt_offset_child - est_offset_child)
    print(f"\nChild Offset (Ground Truth):  {np.round(gt_offset_child, 5)}")
    print(f"Child Offset (Estimated):     {np.round(est_offset_child, 5)}")
    print(f"--> Child Estimation Error (L2 Norm):  {child_error_norm:.6f} meters")
    
    mean_residual = np.mean(np.linalg.norm(residual_error, axis=1))
    print(f"\nMean residual of the least-squares fit: {mean_residual:.6f} meters")
    print("\n" + "="*25)

def demonstrate_hinge_axis_estimation():
    """
    Creates a parent and child trial connected by a 1-DOF hinge joint,
    estimates the axis, and compares to ground truth.
    """
    print("Step 1: Generating a random 'parent' plate trial...")
    parent_plate_trial = PlateTrial.generate_random_plate_trial(
        duration=15, fs=100, add_noise=False
    )
    print("  - Parent trial created.")

    print("\nStep 2: Defining ground truth 1-DOF hinge joint parameters...")
    gt_axis_in_parent = np.array([0.577, 0.577, 0.577]) # An arbitrary axis
    gt_axis_in_parent /= np.linalg.norm(gt_axis_in_parent) # Normalize
    


    print(f"  - Ground Truth Axis (Parent Frame): {np.round(gt_axis_in_parent, 4)}")
    print(f"  - Ground Truth Axis (Child Frame):  {np.round(gt_axis_in_child, 4)}")

    print("\nStep 3: Generating connected 'child' plate trial...")
    child_plate_trial = parent_plate_trial.generate_random_connected_plate_trial(
        dof=1,
        joint_center_parent=np.array([0.2, 0, 0]),
        joint_center_child=np.array([-0.2, 0, 0]),
        axis_parent_local=gt_axis_in_parent, # Pass the ground truth axis
        add_noise=False
    )
    print("  - Child trial generated.")

    print("\nStep 4: Estimating hinge axis using `find_hinge_joint_axis`...")
    gyro_result = parent_plate_trial.imu_trace.find_hinge_joint_axis(
        child_plate_trial.imu_trace, subsample_rate=10
    )
    est_axis_gyro_parent = gyro_result['axis_self']
    est_axis_gyro_child = gyro_result['axis_other']
    print(f"  - Estimation complete. Converged: {gyro_result['converged']}")

    print("\n" + "="*35)
    print("      HINGE AXIS RESULTS")
    print("="*35)

    error_gyro_parent = calculate_axis_error_deg(gt_axis_in_parent, est_axis_gyro_parent)
    print("\n--- Parent Frame Axis ---")
    print(f"Ground Truth:     {np.round(gt_axis_in_parent, 4)}")
    print(f"Estimate (Gyros): {np.round(est_axis_gyro_parent, 4)}")
    print(f"--> Error:        {error_gyro_parent:.4f} degrees")

    error_gyro_child = calculate_axis_error_deg(gt_axis_in_child, est_axis_gyro_child)
    print("\n--- Child Frame Axis ---")
    print(f"Ground Truth:     {np.round(gt_axis_in_child, 4)}")
    print(f"Estimate (Gyros): {np.round(est_axis_gyro_child, 4)}")
    print(f"--> Error:        {error_gyro_child:.4f} degrees")
    print("\n" + "="*35)

def demonstrate_biaxial_joint_axis_estimation():
    """
    Demonstrates estimation of two local-frame axes for a 2-DOF joint.
    """
    print("Step 1: Generating a random 'parent' plate trial...")
    parent_plate_trial = PlateTrial.generate_random_plate_trial(
        duration=15, fs=100, add_noise=False
    )
    print("  - Parent trial created.")

    print("\nStep 2: Defining ground truth biaxial joint parameters...")
    gt_axis_parent_local = np.random.rand(3)
    gt_axis_parent_local /= np.linalg.norm(gt_axis_parent_local)
    gt_axis_child_local = np.random.rand(3)
    gt_axis_child_local /= np.linalg.norm(gt_axis_child_local)
    carrying_angle = np.random.uniform(low=0, high=np.pi) # Between 0 and 180 degrees
    print(f"  - Ground Truth Carrying Angle: {np.rad2deg(carrying_angle):.4f} degrees")
    
    print(f"  - GT Axis (Parent's Local Frame): {np.round(gt_axis_parent_local, 4)}")
    print(f"  - GT Axis (Child's Local Frame):  {np.round(gt_axis_child_local, 4)}")

    print("\nStep 3: Generating connected 'child' trial using the new generator...")
    child_plate_trial = parent_plate_trial.generate_2dof_plate(
        parent_offset=np.array([0.3, 0, 0]),
        child_offset=np.array([0, 0.2, 0]),
        j1_parent=gt_axis_parent_local,
        j2_child=gt_axis_child_local,
        carrying_angle=carrying_angle,
        add_noise=False
    )

    # Double check that the child trial was created
    # Get the relative angular velocity between parent and child
    p_world_frame = parent_plate_trial.get_imu_trace_in_global_frame()
    c_world_frame = child_plate_trial.get_imu_trace_in_global_frame()
    w_rel = [g_c - g_p for g_p, g_c in zip(p_world_frame.gyro, c_world_frame.gyro)]
    p_axis_world = [r @ gt_axis_parent_local for r in parent_plate_trial.world_trace.rotations]
    c_axis_world = [r @ gt_axis_child_local for r in child_plate_trial.world_trace.rotations]
    normal_axis_world = [np.cross(p, c) for p, c in zip(p_axis_world, c_axis_world)]

    w_rel_proj = [np.dot(w, n) for w, n in zip(w_rel, normal_axis_world)]

    plt.figure(figsize=(10, 4))
    plt.plot(w_rel_proj, label='Relative Angular Velocity Projection', color='blue')
    plt.title('Verification of 2-DOF Joint: Relative Angular Velocity Projection')
    plt.xlabel('Sample Index')
    plt.ylabel('Angular Velocity Projection (rad/s)')
    plt.axhline(0, color='red', linestyle='--', label='Zero Line')
    plt.legend()
    plt.grid()
    # plt.show()
    

    print("  - Child trial generated successfully.")

    print("\nStep 4: Estimating joint axes using `find_biaxial_joint_axes`...")
    result = parent_plate_trial.find_biaxial_joint_axes(
        other=child_plate_trial, subsample_rate=1, verbose=True
    )
    est_axis_parent = result['axis_parent_local']
    est_axis_child = result['axis_child_local']
    print(f"  - Estimation complete. Converged: {result['converged']}")

    print("\n" + "="*45)
    print("      BIAXIAL AXIS ESTIMATION RESULTS")
    print("="*45)

    # The algorithm doesn't know which axis is "parent" vs "child", so we must
    # check both possible pairings and choose the one with the lower total error.
    error_A_parent = calculate_axis_error_deg(gt_axis_parent_local, est_axis_parent)
    error_A_child = calculate_axis_error_deg(gt_axis_child_local, est_axis_child)
    total_error_A = error_A_parent + error_A_child

    error_B_parent = calculate_axis_error_deg(gt_axis_child_local, est_axis_parent)
    error_B_child = calculate_axis_error_deg(gt_axis_parent_local, est_axis_child)
    total_error_B = error_B_parent + error_B_child

    parent_axis_world_est = [r @ est_axis_parent for r in parent_plate_trial.world_trace.rotations]
    child_axis_world_est = [r @ est_axis_child for r in child_plate_trial.world_trace.rotations]
    carrying_angles_est = [np.arccos(np.clip(np.dot(c, p), -1.0, 1.0)) for p, c in zip(parent_axis_world_est, child_axis_world_est)]
    angle_variation_est = np.rad2deg(np.std(carrying_angles_est))
    carrying_angle_est = np.mean(carrying_angles_est)
    print(f"  - Verified carrying angle constancy: Std Dev = {angle_variation_est:.6f} degrees, Mean = {np.rad2deg(carrying_angle_est):.4f} degrees")

    if total_error_A < total_error_B:
        print("\n--- Parent's Local Axis ---")
        print(f"Ground Truth: {np.round(gt_axis_parent_local, 4)}")
        print(f"Estimated:    {np.round(est_axis_parent, 4)}")
        print(f"--> Error:    {error_A_parent:.4f} degrees")
        
        print("\n--- Child's Local Axis ---")
        print(f"Ground Truth: {np.round(gt_axis_child_local, 4)}")
        print(f"Estimated:    {np.round(est_axis_child, 4)}")
        print(f"--> Error:    {error_A_child:.4f} degrees")

        print(f"\n--- Carrying Angle ---")
        print(f"Ground Truth: {np.rad2deg(carrying_angle):.4f} degrees")
        print(f"Estimated:    {np.rad2deg(carrying_angle_est):.4f} degrees")
        print(f"--> Error:    {np.rad2deg(np.abs(carrying_angle - carrying_angle_est)):.4f} degrees")

    else:
        print("\n--- Parent's Local Axis (Matched to GT Child) ---")
        print(f"Ground Truth: {np.round(gt_axis_child_local, 4)}")
        print(f"Estimated:    {np.round(est_axis_parent, 4)}")
        print(f"--> Error:    {error_B_parent:.4f} degrees")
        
        print("\n--- Child's Local Axis (Matched to GT Parent) ---")
        print(f"Ground Truth: {np.round(gt_axis_parent_local, 4)}")
        print(f"Estimated:    {np.round(est_axis_child, 4)}")
        print(f"--> Error:    {error_B_child:.4f} degrees")

    print("\n" + "="*45)

if __name__ == '__main__':
    # demonstrate_joint_center_estimation()
    # print("\n\n" + "#"*50 + "\n\n")
    # demonstrate_hinge_axis_estimation()
    # print("\n\n" + "#"*50 + "\n\n")
    demonstrate_biaxial_joint_axis_estimation()