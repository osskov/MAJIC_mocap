"""
Synthetic traces for tests.

These build PlateTrials and WorldTraces out of nothing — no file, no dataset, no format — so
they are fixtures rather than data loading, and they do not belong on the trace classes or in
the build package. They used to be ~350 lines of test scaffolding sitting on PlateTrial and
WorldTrace alongside the real physics.

`generate_1dof_plate`, `generate_2dof_plate` and `generate_3dof_plate` take the parent plate
as their first argument; they were instance methods.
"""
import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace


# Marker layout of a plate, in the plate's own frame. This has to agree with the
# convention _reconstruct_from_markers reconstructs against: +x runs d->x and y->o,
# +y runs x->o and d->y, and the four markers are symmetric about the plate origin.
#
# Shared rather than duplicated because TestWorldTrace and TestReconstruction both build
# synthetic plates from it, and a plate that disagreed between them would make one of the
# two silently test a different geometry than it claims to.
PLATE_HALF_WIDTH = 0.04
PLATE_HALF_HEIGHT = 0.03
MARKER_O_LOCAL = np.array([PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0])
MARKER_D_LOCAL = np.array([-PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0])
MARKER_X_LOCAL = np.array([PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0])
MARKER_Y_LOCAL = np.array([-PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0])


def markers_from_poses(positions, rotations):
    """Places the four plate markers in the world, given the plate's pose over time."""
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    return tuple(
        np.einsum('nij,j->ni', rotations, local) + positions
        for local in (MARKER_O_LOCAL, MARKER_D_LOCAL, MARKER_X_LOCAL, MARKER_Y_LOCAL)
    )


def require_data(condition: bool, what: str) -> None:
    """FAIL when the source data or built cache is missing, unless told to skip.

    The strongest tests in this suite need `data/` (a large external download) or a built
    parquet cache, and they used to skip themselves when either was absent. That is the right
    instinct locally and the wrong outcome everywhere else: a clean checkout ran a materially
    weaker suite and still reported green, so "all tests pass" meant one thing on a machine
    with the data and something much weaker on CI or a collaborator's first checkout.

    Failing by default makes the gap visible. `MAJIC_ALLOW_MISSING_DATA=1` restores skipping
    for the case where someone genuinely cannot download 40 GB and wants the rest of the
    suite -- an explicit, recorded choice rather than a silent default.
    """
    if condition:
        return
    if os.environ.get('MAJIC_ALLOW_MISSING_DATA'):
        raise unittest.SkipTest(f"{what} (MAJIC_ALLOW_MISSING_DATA is set)")
    raise AssertionError(
        f"{what}.\n"
        f"These tests exercise the real build path and cannot verify it without the data. "
        f"Fetch it (see README) and run `python -m experiments.build_trials --dataset "
        f"alborno`, or set MAJIC_ALLOW_MISSING_DATA=1 to skip them and accept a weaker suite."
    )


def generate_smooth_motion_profile(
        num_samples: int,
        duration: float,
        num_waves: int = 4,
        max_amp: float = 1.0
    ) -> np.ndarray:
        """
        Generates a smooth, complex 1D motion profile using a sum of sine waves.

        Args:
            num_samples (int): The number of data points to generate.
            duration (float): The total time duration in seconds.
            num_waves (int): The number of sine waves to sum for complexity.
            max_amp (float): The maximum amplitude of the resulting motion.

        Returns:
            np.ndarray: A 1D array representing the motion profile.
        """
        t = np.linspace(0, duration, num_samples, endpoint=False)
        motion = np.zeros(num_samples)
        for i in range(1, num_waves + 1):
            amplitude = np.random.uniform(0.1, 1.0) * max_amp / num_waves
            frequency = np.random.uniform(0.1, 2.0) * i
            phase = np.random.uniform(0, 2 * np.pi)
            motion += amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return motion

def generate_random_world_trace(duration: float = 10.0, fs: float = 100.0) -> 'WorldTrace':
    """
    Generates a WorldTrace with random but smooth position and orientation.

    Args:
        duration (float): The duration of the trial in seconds.
        fs (float): The sampling frequency in Hz.

    Returns:
        WorldTrace: The generated world trace.
    """
    num_samples = int(duration * fs)
    timestamps = np.linspace(0, duration, num_samples, endpoint=False)

    # --- Generate smooth random position ---
    pos_x = generate_smooth_motion_profile(num_samples, duration, max_amp=0.5)
    pos_y = generate_smooth_motion_profile(num_samples, duration, max_amp=0.3)
    pos_z = generate_smooth_motion_profile(num_samples, duration, max_amp=0.5)
    positions = np.column_stack((pos_x, pos_y, pos_z))

    # --- Generate smooth random orientation ---
    # Create motion profiles for Euler angles
    rot_z = generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi)
    rot_y = generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi / 2)
    rot_x = generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi / 2)

    # Convert Euler angles to a stack of rotation matrices
    rotations_obj = Rotation.from_euler('zyx', np.vstack([rot_z, rot_y, rot_x]).T)
    rotations = rotations_obj.as_matrix()

    return WorldTrace(timestamps, positions, rotations)

def generate_random_plate_trial(
    duration: float = 10.0,
    fs: float = 100.0,
    add_noise: bool = True,
    gyro_noise_std: float = 0.005,
    acc_noise_std: float = 0.05
) -> 'PlateTrial':
    """
    Generates a single PlateTrial with random but smooth motion.

    This function performs the following steps:
    1. Creates a smooth, random 3D position and orientation trajectory (WorldTrace).
    2. Calculates the ideal IMU data (gyroscope, accelerometer) that corresponds
    to this trajectory.
    3. Adds synthetic Gaussian noise to the IMU data to simulate a real sensor.
    4. Combines the world and IMU traces into a single PlateTrial object.

    Args:
        duration (float, optional): The duration of the trial in seconds. Defaults to 10.0.
        fs (float, optional): The sampling frequency in Hz. Defaults to 100.0.
        add_noise (bool, optional): If True, adds noise to the synthetic IMU data.
            Defaults to True.
        gyro_noise_std (float, optional): Standard deviation of the gyroscope
            noise in rad/s. Defaults to 0.005.
        acc_noise_std (float, optional): Standard deviation of the accelerometer
            noise in m/s^2. Defaults to 0.05.

    Returns:
        PlateTrial: A new PlateTrial object with synthetic data.
    """
    # 1. Generate the ground-truth WorldTrace
    world_trace = generate_random_world_trace(duration, fs)

    # 2. Calculate the corresponding "perfect" IMU trace
    # Define a standard gravity vector
    gravity = np.array([0, 0, -9.81])
    magnetic_field = np.array([0.2, 0, 0.4])  # Example magnetic field vector
    imu_trace = world_trace.calculate_imu_trace(acc_from_gravity=gravity, magnetic_field=magnetic_field)

    # 3. (Optional) Add realistic noise to the IMU data
    if add_noise:
        imu_trace = imu_trace.add_noise(gyro_noise_std, acc_noise_std)

    # 4. Create and return the final PlateTrial object
    return PlateTrial(name="synthetic_random_trial", imu_trace=imu_trace, world_trace=world_trace)

def generate_1dof_plate(
    parent: PlateTrial,
    joint_center_parent: np.ndarray,
    joint_center_child: np.ndarray,
    parent_to_joint_rotation: Rotation = None,
    child_to_joint_rotation: Rotation = None,
    add_noise: bool = True,
    gyro_noise_std: float = 0.005,
    acc_noise_std: float = 0.05
) -> 'PlateTrial':
    """
    Generates a child PlateTrial connected by a 1-DOF hinge joint.
    The kinematic chain is:
    R_child = R_parent @ R_p2j @ R_joint_motion(z) @ R_c2j.inv()
    """
    if parent_to_joint_rotation is None: parent_to_joint_rotation = Rotation.random()
    if child_to_joint_rotation is None: child_to_joint_rotation = Rotation.random()

    parent_world_trace = parent.world_trace
    num_samples = len(parent_world_trace)
    duration = parent_world_trace.timestamps[-1] - parent_world_trace.timestamps[0]

    angle = generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi)
    # angle[:, None], not angle. scipy 1.17 requires the last dimension to match the number
    # of sequence axes, so from_euler('z', (N,)) now raises where it used to broadcast.
    R_joint_motion = Rotation.from_euler('z', angle[:, None])

    R_parent_matrices = parent.world_trace.rotations
    R_p2j_mat = parent_to_joint_rotation.as_matrix()
    R_c2j_inv_mat = child_to_joint_rotation.as_matrix().T
    R_rel_total_matrices = R_p2j_mat @ R_joint_motion.as_matrix() @ R_c2j_inv_mat
    R_child_matrices = R_parent_matrices @ R_rel_total_matrices

    P_parent = parent.world_trace.positions
    parent_offset_global = (R_parent_matrices @ joint_center_parent).squeeze()
    child_offset_global = (R_child_matrices @ joint_center_child).squeeze()
    P_child = P_parent + parent_offset_global - child_offset_global

    child_world_trace = WorldTrace(
        timestamps=parent_world_trace.timestamps,
        positions=[row for row in P_child],
        rotations=[mat for mat in R_child_matrices]
    )

    gravity = np.array([0, 0, -9.81])
    child_imu_trace = child_world_trace.calculate_imu_trace(acc_from_gravity=gravity)
    if add_noise:
        child_imu_trace = child_imu_trace.add_noise(gyro_noise_std, acc_noise_std)

    return PlateTrial(f"{parent.name}_child_dof1", child_imu_trace, child_world_trace)


def generate_2dof_plate(
    parent: PlateTrial,
    j1_parent: np.ndarray,
    j2_child: np.ndarray,
    carrying_angle: float,
    parent_offset: np.ndarray,
    child_offset: np.ndarray,
    add_noise: bool = True,
    gyro_noise_std: float = 0.005,
    acc_noise_std: float = 0.05
) -> 'PlateTrial':
    """
    Generates a child PlateTrial connected by a 2-DOF joint
    with a fixed carrying angle.

    The kinematic model is:
    R_wc = R_wp @ R_pj1 @ R_j1j2 @ R_j2c

    Where:
    - R_wp: Parent's world rotation.
    - R_pj1: Rotation aligning the parent-frame axis (j1_parent) to [0,0,1] (Z-axis).
    - R_j1j2: The ZYX joint rotation [z_angle, carrying_angle, x_angle].
    - R_j2c: Rotation aligning the child-frame axis (j2_child) to [1,0,0] (X-axis).
    """
    parent_world_trace = parent.world_trace
    num_samples = len(parent_world_trace)
    duration = parent_world_trace.timestamps[-1] - parent_world_trace.timestamps[0]
    timestamps = parent_world_trace.timestamps

    # === 1. Calculate Constant Alignment Rotations ===
    
    # Normalize input axes
    j1_p_norm = j1_parent / np.linalg.norm(j1_parent)
    j2_c_norm = j2_child / np.linalg.norm(j2_child)

    z_axis = np.array([0., 0., 1.])
    x_axis = np.array([1., 0., 0.])

    # R_pj1: "aligns [0,0,1] to j1_parent"
    # This finds R such that R @ z_axis = j1_p_norm
    # align_vectors(target, source)
    R_pj1_rot = Rotation.align_vectors(j1_p_norm[np.newaxis, :], z_axis[np.newaxis, :])[0]
    R_pj1_mat = R_pj1_rot.as_matrix()

    # R_j2c: "aligns j2_child to [1,0,0]"
    # This finds R such that R @ j2_c_norm = x_axis
    R_j2c_rot = Rotation.align_vectors(x_axis[np.newaxis, :], j2_c_norm[np.newaxis, :])[0]
    R_j2c_mat = R_j2c_rot.as_matrix()

    # === 2. Generate Joint Motion Profile (R_j1j2) ===
    
    # Generate simple motion for the Z and X axes
    z_angles = generate_smooth_motion_profile(num_samples=num_samples,duration=duration, max_amp=np.pi/2) # Flexion/Extension
    x_angles = generate_smooth_motion_profile(num_samples=num_samples,duration=duration, max_amp=np.pi/6) # Abduction/Adduction
    
    # Convert to a (N, 3, 3) stack of rotation matrices
    R_j1 = Rotation.from_euler('z', z_angles[:, None])
    R_carrying = Rotation.from_euler('y', (carrying_angle * np.ones(num_samples))[:, None])
    R_j2 = Rotation.from_euler('x', x_angles[:, None])

    R_j1j2_mat = (R_j1 * R_carrying * R_j2).as_matrix()

    # === 3. Calculate Child World Rotations ===
    R_wp = parent.world_trace.rotations
    
    # R_child = R_wp @ R_pj1 @ R_j1j2 @ R_j2c
    # (N,3,3) = (N,3,3) @ (3,3) @ (N,3,3) @ (3,3)
    R_wc = R_wp @ R_pj1_mat @ R_j1j2_mat @ R_j2c_mat

    # === 4. Calculate Child World Positions ===
    P_parent = parent.world_trace.positions

    # Apply parent rotation to parent offset vector (for all N samples)
    # 'nij,j->ni' means: (N, 3, 3) @ (3,) -> (N, 3)
    parent_offset_global = np.einsum('nij,j->ni', R_wp, parent_offset)
    
    # Apply child rotation to child offset vector (for all N samples)
    child_offset_global = np.einsum('nij,j->ni', R_wc, child_offset)

    # Child position = Parent pos + Parent offset - Child offset
    P_child = P_parent + parent_offset_global - child_offset_global

    # === 5. Create Traces and Return PlateTrial ===
    child_world_trace = WorldTrace(
        timestamps=parent_world_trace.timestamps,
        positions=[row for row in P_child],
        rotations=[mat for mat in R_wc]
    )

    gravity = np.array([0, 0, -9.81])
    child_imu_trace = child_world_trace.calculate_imu_trace(acc_from_gravity=gravity)
    
    if add_noise:
        child_imu_trace = child_imu_trace.add_noise(gyro_noise_std, acc_noise_std)

    # # === Verification of Relative Angular Velocity ===
    
    # # Get actual relative angular velocity from IMU data (in world frame)
    # w_c_world = np.array([r @ g for r, g in zip(child_world_trace.rotations, child_imu_trace.gyro)])
    # w_p_world = np.array([r @ g for r, g in zip(parent.world_trace.rotations, parent.imu_trace.gyro)])
    # w_rel_actual = w_c_world - w_p_world
    # print(w_rel_actual[50])
    # # Calculate ideal relative angular velocity from joint angle derivatives
    
    # # Use np.gradient for a stable derivative that matches array length
    # dt = np.mean(np.diff(timestamps))
    # z_dot = (np.roll(z_angles, -1) - z_angles) / dt
    # x_dot = (np.roll(x_angles, -1) - x_angles) / dt

    # # Fix the last sample (which was wrapped around)
    # z_dot[-1] = z_dot[-2]
    # x_dot[-1] = x_dot[-2]

    # # y_dot is zero, so we omit it
    
    # # --- Component 1: z_dot around parent's j1 axis (in world frame) ---
    # # This is the Z-axis in the J1 frame ([0,0,1]), rotated by R_wp @ R_pj1
    # # The axis in world frame is R_wp @ j1_p_norm
    # w_axis_j1_world = np.einsum('nij,j->ni', R_wp, j1_p_norm)
    # w_rel_1_world = z_dot[:, np.newaxis] * w_axis_j1_world
    # print(w_axis_j1_world[50])
    # # --- Component 2: x_dot around the ZYX sequence's X-axis (in world frame) ---
    # # This is the X-axis ([1,0,0]) *after* the Rz and Ry rotations.
    # # Its orientation in the world is:
    # # R_wp @ R_pj1 @ R_z(t) @ R_y(t) @ [1,0,0]
    
    # w_axis_j2_world = np.einsum('nij,j->ni', R_wc, j2_c_norm)
    # w_rel_2_world = x_dot[:, np.newaxis] * w_axis_j2_world
    # print(w_axis_j2_world[50])
    # # Ideal total relative velocity is the sum of the two components
    # w_rel_ideal = w_rel_1_world + w_rel_2_world
    
    # # Compare the actual (from IMU) vs ideal (from joint angles)
    # # We skip the first few samples to avoid gradient artifacts at edges
    # skip = 5 
    # error = np.linalg.norm(w_rel_actual[skip:-skip] - w_rel_ideal[skip:-skip], axis=1)
    
    # print(f"2-DOF Joint Gen: Mean w_rel (actual vs. ideal) error: {np.mean(error):.6f} rad/s")

    # j3_world = np.cross(w_axis_j1_world, w_axis_j2_world)
    # dot_products = np.einsum('ni,ni->n', w_rel_actual, j3_world)
    # print(f"2-DOF Joint Gen: Mean dot(w_rel_actual, j1 x j2): {np.mean(dot_products):.6f} (should be near 0)")
    # dot_products_ideal = np.einsum('ni,ni->n', w_rel_ideal, j3_world)
    # print(f"2-DOF Joint Gen: Mean dot(w_rel_ideal, j1 x j2): {np.mean(dot_products_ideal):.6f} (should be near 0)")

    # # The magnitude of the cross product of joint axes in the real world should also be constant
    # j3 = np.cross(w_axis_j1_world, w_axis_j2_world)
    # j3_magnitudes = np.linalg.norm(j3, axis=1)
    # print(f"2-DOF Joint Gen: Joint axes cross-product magnitude (should be constant): "
    #       f"mean={np.mean(j3_magnitudes):.6f}, std={np.std(j3_magnitudes):.6f}")
    # # --- End of new code block --
    return PlateTrial(f"{parent.name}_child_dof2", child_imu_trace, child_world_trace)

def generate_3dof_plate(
    parent: PlateTrial,
    joint_center_parent: np.ndarray,
    joint_center_child: np.ndarray,
    parent_to_joint_rotation: Rotation = None,
    child_to_joint_rotation: Rotation = None,
    add_noise: bool = True,
    gyro_noise_std: float = 0.005,
    acc_noise_std: float = 0.05
) -> 'PlateTrial':
    """
    Generates a child PlateTrial connected by a 3-DOF spherical joint.
    The kinematic chain is:
    R_child = R_parent @ R_p2j @ R_joint_motion(zyx) @ R_c2j.inv()
    """
    if parent_to_joint_rotation is None: parent_to_joint_rotation = Rotation.random()
    if child_to_joint_rotation is None: child_to_joint_rotation = Rotation.random()

    parent_world_trace = parent.world_trace
    num_samples = len(parent_world_trace)
    duration = parent_world_trace.timestamps[-1] - parent_world_trace.timestamps[0]

    angles = [generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi/2) for _ in range(3)]
    R_joint_motion = Rotation.from_euler('zyx', np.vstack(angles).T)

    R_parent_matrices = parent.world_trace.rotations
    R_p2j_mat = parent_to_joint_rotation.as_matrix()
    R_c2j_inv_mat = child_to_joint_rotation.as_matrix().T
    R_rel_total_matrices = R_p2j_mat @ R_joint_motion.as_matrix() @ R_c2j_inv_mat
    R_child_matrices = R_parent_matrices @ R_rel_total_matrices

    P_parent = parent.world_trace.positions
    parent_offset_global = (R_parent_matrices @ joint_center_parent).squeeze()
    child_offset_global = (R_child_matrices @ joint_center_child).squeeze()
    P_child = P_parent + parent_offset_global - child_offset_global

    child_world_trace = WorldTrace(
        timestamps=parent_world_trace.timestamps,
        positions=[row for row in P_child],
        rotations=[mat for mat in R_child_matrices]
    )

    gravity = np.array([0, 0, -9.81])
    child_imu_trace = child_world_trace.calculate_imu_trace(acc_from_gravity=gravity)
    if add_noise:
        child_imu_trace = child_imu_trace.add_noise(gyro_noise_std, acc_noise_std)

    return PlateTrial(f"{parent.name}_child_dof3", child_imu_trace, child_world_trace)
