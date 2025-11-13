import os
import numpy as np
import scipy.signal as signal
from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace, _generate_smooth_motion_profile
from typing import Tuple, List, Dict, Union
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm  # Import tqdm for progress bars

IMU_TO_TRC_NAME_MAP = {
    'pelvis_imu': 'Pelvis_IMU', 'femur_r_imu': 'R.Femur_IMU', 'femur_l_imu': 'L.Femur_IMU',
    'tibia_r_imu': 'R.Tibia_IMU', 'tibia_l_imu': 'L.Tibia_IMU', 'calcn_r_imu': 'R.Foot_IMU',
    'calcn_l_imu': 'L.Foot_IMU', 'torso_imu': 'Back_IMU'
}

class PlateTrial:
    """
    Contains a synchronized time-series of motion data for a single rigid body.
    
    A "Plate" represents a rigid body (e.g., a segment of a limb) that is
    tracked by both an IMU sensor and an external motion capture system (like
    Vicon or OptiTrack). This class stores the synchronized data from both
    sources.

    Attributes:
        name (str): A descriptive name for the plate (e.g., 'femur_r_imu').
        imu_trace (IMUTrace): An object containing the time-series data from the
            IMU sensor (accelerometer, gyroscope, magnetometer) in the
            sensor's local coordinate frame.
        world_trace (WorldTrace): An object containing the time-series data from
            the motion capture system (position, orientation) in the
            global coordinate frame.
    """

    name: str
    imu_trace: IMUTrace
    world_trace: WorldTrace

    def __init__(self, name: str, imu_trace: IMUTrace, world_trace: WorldTrace):
        """
        Initializes a new PlateTrial object.

        Args:
            name (str): The name for this plate trial.
            imu_trace (IMUTrace): The IMU data trace.
            world_trace (WorldTrace): The World (mocap) data trace.

        Raises:
            AssertionError: If the IMU and World traces have different lengths
                or their timestamps do not match within a small tolerance (1e-8).
        """
        # Ensure data consistency
        assert len(imu_trace) == len(world_trace), \
            "IMU and World traces must have the same length"
        
        # Check for timestamp synchronization
        if max(np.abs(imu_trace.timestamps - world_trace.timestamps)) > 1e-8:
            print(f"Warning: IMU and World traces must have the same timestamps. "
                  f"Max difference: {max(np.abs(imu_trace.timestamps - world_trace.timestamps))}")
        
        assert max(np.abs(imu_trace.timestamps - world_trace.timestamps)) < 1e-8, \
            "Timestamps must match"
        
        # Type checking
        assert isinstance(imu_trace, IMUTrace)
        assert isinstance(world_trace, WorldTrace)

        self.name = name
        self.imu_trace = imu_trace
        self.world_trace = world_trace


    def __len__(self) -> int:
        """Returns the number of samples (timesteps) in the trial."""
        return len(self.imu_trace)

    def __getitem__(self, key: slice) -> 'PlateTrial':
        """
        Enables slicing of the PlateTrial object.

        This allows you to take a subset of the time-series data, e.g., `trial[100:500]`.

        Args:
            key (slice): The slice object specifying the start, stop, and step.

        Returns:
            PlateTrial: A new PlateTrial object containing only the sliced data.
        """
        if not isinstance(key, slice):
            raise TypeError("PlateTrial slicing only supports slice objects.")
        return PlateTrial(self.name, self.imu_trace.copy()[key], self.world_trace.copy()[key])
    
    def copy(self) -> 'PlateTrial':
        """
        Returns a deep copy of the PlateTrial object.

        This creates new IMUTrace and WorldTrace objects with copied data,
        preventing modifications to the copy from affecting the original.

        Returns:
            PlateTrial: A new, independent copy of the object.
        """
        return PlateTrial(
            self.name,
            self.imu_trace.copy(),
            self.world_trace.copy()
        )

    def _align_world_trace_to_imu_trace(self) -> 'PlateTrial':
        """
        Aligns the WorldTrace's orientation to the IMUTrace's orientation.

        This method calculates the static rotational offset between the "ground truth"
        angular velocity (from WorldTrace) and the measured angular velocity
        (from IMUTrace). It then applies this offset to the WorldTrace's
        orientation data so that the coordinate frames are aligned.

        This is a crucial step for sensor-to-segment calibration.

        Returns:
            PlateTrial: A new PlateTrial object with the aligned WorldTrace.
        """
        # 1. Calculate a synthetic IMU gyro trace from the world (mocap) rotations.
        synthetic_imu_trace = self.world_trace.calculate_imu_trace(skip_lin_acc=True)
        
        # 2. Find the rotation (R_wt_it) that maps the world trace to the imu trace
        #    by comparing their gyroscope data.
        R_wt_it = synthetic_imu_trace.calculate_rotation_offset_from_gyros(self.imu_trace)
        
        # 3. Apply this static rotation to all orientations in the world trace.
        #    new_R_world = old_R_world @ R_wt_it
        new_world_rotations = [rot @ R_wt_it for rot in self.world_trace.rotations]
        
        # 4. Create a new WorldTrace and PlateTrial with the aligned data.
        new_world_trace = WorldTrace(self.world_trace.timestamps, self.world_trace.positions, new_world_rotations)
        return PlateTrial(self.name, self.imu_trace, new_world_trace)

    def project_imu_trace(self, local_offset: np.ndarray) -> IMUTrace:
        r"""
        Estimates the IMUTrace values at a different location on the same rigid body.
        Currently only projects accelerometer data, but could be extended to gyro/mag.

        This uses the rigid body equations of motion to project the accelerometer
        data to a new point, defined by the `local_offset` vector in the
        IMU's coordinate frame.
        
        $a_p = a_c + \alpha \times r + \omega \times (\omega \times r)$
        
        where:
        - $a_c$ is the measured acceleration at the IMU center.
        - $\omega$ is the measured angular velocity.
        - $\alpha$ is the angular acceleration (derived from gyro).
        - $r$ is the `local_offset` vector (from IMU center to new point P).
        - $a_p$ is the projected acceleration at point P.

        Args:
            local_offset (np.ndarray): A (3,) array representing the 3D
                vector from the current IMU location to the new,
                projected location, expressed in the IMU's local frame.

        Returns:
            IMUTrace: A new IMUTrace object with the projected sensor data.
        """
        # This function relies on the IMUTrace class to handle the physics.
        return self.imu_trace.project_acc(local_offset)
    
    @staticmethod
    def generate_plate_from_traces(
        imu_traces: Dict[str, 'IMUTrace'], 
        world_traces: Dict[str, 'WorldTrace'], 
        align_plate_trials: bool
    ) -> List['PlateTrial']:
        """
        Factory method to create a list of PlateTrial objects from raw data.

        This function performs the key "data wrangling" steps:
        1.  Finds matching pairs of IMU and World traces (using IMU_TO_TRC_NAME_MAP).
        2.  Resamples IMU data if its frequency doesn't match the World trace.
        3.  Synchronizes the traces in time using cross-correlation of gyro norms.
        4.  (Optionally) Aligns the coordinate frames.
        5.  Trims all resulting trials to the same minimum length for consistency.

        Args:
            imu_traces (Dict[str, 'IMUTrace']): A dictionary mapping IMU names
                to IMUTrace objects.
            world_traces (Dict[str, 'WorldTrace']): A dictionary mapping segment
                names to WorldTrace objects.
            align_plate_trials (bool): If True, performs sensor-to-segment
                alignment using `_align_world_trace_to_imu_trace`.

        Returns:
            List['PlateTrial']: A list of processed, synchronized, and aligned
                PlateTrial objects.
        """
        plate_trials = []
        imu_slice, world_slice = slice(0, 0), slice(0, 0)
        
        if not imu_traces:
            print("Warning: No IMU traces loaded.")
            return []
        
        # --- ADDED TQDM ---
        # Wrap the imu_traces dictionary items with tqdm for a progress bar
        print("Processing and synchronizing traces...")
        for imu_name, imu_trace in tqdm(imu_traces.items(), desc="Generating PlateTrials"):
            try:
                # Find the corresponding world_trace, using the name map as a fallback
                world_trace = world_traces[imu_name] if imu_name in world_traces \
                    else world_traces[IMU_TO_TRC_NAME_MAP[imu_name]]
            except KeyError:
                print(f"IMU {imu_name} not found in world traces. Skipping.")
                continue

            # Resample if frequencies don't match (within a small tolerance)
            if abs(imu_trace.get_sample_frequency() - world_trace.get_sample_frequency()) > 0.2:
                # print(f"Sample frequency mismatch for {imu_name}: IMU {imu_trace.get_sample_frequency()} Hz, World {world_trace.get_sample_frequency()} Hz")
                imu_trace = imu_trace.resample(float(world_trace.get_sample_frequency()))

            # Sync traces by finding the optimal time lag
            imu_slice, world_slice = PlateTrial._sync_traces(imu_trace, world_trace)
            
            # Create new traces based on the synchronized slices
            synced_imu_trace = imu_trace[imu_slice].re_zero_timestamps()
            synced_world_trace = world_trace[world_slice].re_zero_timestamps()
            
            # Create the new PlateTrial object
            new_plate_trial = PlateTrial(imu_name, synced_imu_trace, synced_world_trace)
            
            # Optionally align the coordinate frames
            if align_plate_trials:
                new_plate_trial = new_plate_trial._align_world_trace_to_imu_trace()
                
            plate_trials.append(new_plate_trial)

        # Ensure all trials have the same length by trimming to the shortest one
        plate_trial_lengths = [len(plate_trial) for plate_trial in plate_trials]
        if len(set(plate_trial_lengths)) > 1:
            print(f"Warning: Plate trials have different lengths: {plate_trial_lengths}. "
                  "Trimming to minimum length.")
            min_length = min(plate_trial_lengths)
            plate_trials = [plate_trial[:min_length] for plate_trial in plate_trials]

        print(f"Successfully generated {len(plate_trials)} PlateTrials.")
        return plate_trials

    @staticmethod
    def load_trial_from_Al_Borno_folder(folder_path: str, align_plate_trials=True) -> List['PlateTrial']:
        """
        Loads a trial from the "Al Borno" dataset structure.

        Expected folder structure:
        - [folder_path]/IMU/ (Contains IMU .csv files)
        - [folder_path]/Mocap/ (Contains motion capture .trc file)

        Args:
            folder_path (str): The root path to the trial folder.
            align_plate_trials (bool): If True, aligns sensor and segment frames.

        Returns:
            List['PlateTrial']: A list of processed PlateTrial objects.
        """
        # 1. Load IMU Traces
        imu_folder = os.path.join(folder_path, 'IMU')
        imu_traces = IMUTrace.load_IMUTraces_from_Al_Borno_folder(imu_folder)

        # 2. Load World Traces
        mocap_folder = os.path.join(folder_path, 'Mocap/')
        # Find the first .trc file that is not a 'static' file
        trc_file_path = [file for file in os.listdir(mocap_folder) if file.endswith('.trc') and 'static' not in file][0]
        trc_file_path = os.path.abspath(os.path.join(mocap_folder, trc_file_path))
        world_traces = WorldTrace.load_from_trc_file(trc_file_path)

        print(f"Loaded {len(imu_traces)} IMU traces and {len(world_traces)} World traces.")

        # 3. Process all traces using the main generation function
        #    (Progress bar will be shown inside this function)
        plate_trials = PlateTrial.generate_plate_from_traces(
            imu_traces, world_traces, align_plate_trials
        )

        return plate_trials
    
    @staticmethod
    def load_trial_from_folder(folder_path: str, align_plate_trials=True) -> List['PlateTrial']:
        """
        Loads a trial from the "Skov" dataset structure (inferred name).

        Expected folder structure:
        - [folder_path]/imu data/ (Contains IMU files)
        - [folder_path]/ (Contains motion capture .trc file in the root)

        Args:
            folder_path (str): The root path to the trial folder.
            align_plate_trials (bool): If True, aligns sensor and segment frames.

        Returns:
            List['PlateTrial']: A list of processed PlateTrial objects.
        """
        # 1. Load IMU Traces
        imu_traces = IMUTrace.load_IMUTraces_from_Skov_folder(folder_path)

        # 2. Load World Traces
        trc_file_path = [file for file in os.listdir(folder_path) if file.endswith('.trc')][0]
        trc_file_path = os.path.abspath(os.path.join(folder_path, trc_file_path))
        world_traces = WorldTrace.load_from_trc_file(trc_file_path)

        print(f"Loaded {len(imu_traces)} IMU traces and {len(world_traces)} World traces.")

        # 3. Process all traces using the main generation function
        #    (Progress bar will be shown inside this function)
        plate_trials = PlateTrial.generate_plate_from_traces(
            imu_traces, world_traces, align_plate_trials
        )
        
        return plate_trials
    
    @staticmethod
    def _sync_traces(imu_trace: IMUTrace, world_trace: WorldTrace) -> Tuple[slice, slice]:
        """
        Synchronizes an IMU trace and a World trace using gyro data.

        This method resamples if necessary, then uses `_sync_arrays` on the
        norm of the gyroscope data (real and synthetic) to find the
        time lag.

        Args:
            imu_trace (IMUTrace): The IMU data.
            world_trace (WorldTrace): The Mocap data.

        Returns:
            Tuple[slice, slice]: A pair of slice objects (imu_slice, world_slice)
                that, when applied to their respective traces, will align them
                in time.
        """
        # Ensure sample frequencies are compatible before syncing
        if not np.isclose(imu_trace.get_sample_frequency(), world_trace.get_sample_frequency(), rtol=0.2):
            imu_trace = imu_trace.resample(float(world_trace.get_sample_frequency()))

        # Calculate a synthetic gyro trace from the world trace
        synthetic_imu_trace = world_trace.calculate_imu_trace(skip_lin_acc=True)
        
        # Sync based on the magnitude (norm) of the angular velocity vectors
        imu_slice, world_slice = PlateTrial._sync_arrays(
            np.linalg.norm(imu_trace.gyro, axis=1),
            np.linalg.norm(synthetic_imu_trace.gyro, axis=1)
        )
        return imu_slice, world_slice

    @staticmethod
    def _sync_arrays(array1: np.ndarray, array2: np.ndarray) -> Tuple[slice, slice]:
        """
        Finds the optimal lag between two 1D arrays using cross-correlation.

        Args:
            array1 (np.ndarray): The first 1D array.
            array2 (np.ndarray): The second 1D array.

        Returns:
            Tuple[slice, slice]: A pair of slice objects (slice1, slice2)
                that trim the arrays to their overlapping, synchronized portions.
        """
        assert array1.ndim == array2.ndim == 1, "Input arrays must be 1D"
        
        # Pad the shorter array to match the longer one for correlation
        max_len = max(len(array1), len(array2))
        a1 = np.pad(array1, (0, max_len - len(array1)), mode='constant')
        a2 = np.pad(array2, (0, max_len - len(array2)), mode='constant')

        # Compute the full cross-correlation
        correlation = signal.correlate(a1, a2, mode='full')
        
        # Find the index of the peak correlation.
        # The lag is this index offset by (max_len - 1)
        lag = np.argmax(correlation) - (max_len - 1)

        # Calculate the start indices and new length for slicing
        # If lag is positive, array1 starts later (trim its start)
        # If lag is negative, array2 starts later (trim its start)
        i1 = max(0, lag)
        i2 = max(0, -lag)
        new_len = min(len(array1) - i1, len(array2) - i2)

        # Return the slice objects
        return slice(i1, i1 + new_len), slice(i2, i2 + new_len)
    
    def get_imu_trace_in_global_frame(self) -> IMUTrace:
        """
        Rotates the IMU sensor data into the global coordinate frame.

        This uses the (ground truth) `world_trace.rotations` to transform
        the accelerometer, gyroscope, and magnetometer vectors from the
        sensor's local frame to the global frame at each timestep.

        Returns:
            IMUTrace: A new IMUTrace object where all data vectors are
                expressed in the global coordinate frame.
        """
        # R is the rotation from local-to-global (from world_trace)
        # v_global = R @ v_local
        rotated_acc = [r @ a for r, a in zip(self.world_trace.rotations, self.imu_trace.acc)]
        rotated_gyro = [r @ g for r, g in zip(self.world_trace.rotations, self.imu_trace.gyro)]
        rotated_mag = [r @ m for r, m in zip(self.world_trace.rotations, self.imu_trace.mag)]
        
        return IMUTrace(
            timestamps=self.imu_trace.timestamps,
            acc=rotated_acc,
            gyro=rotated_gyro,
            mag=rotated_mag
        )
    
    @staticmethod
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
        world_trace = WorldTrace.generate_random_world_trace(duration, fs)

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
    
    def find_biaxial_joint_axes(
        self: 'PlateTrial',
        other: 'PlateTrial',
        initial_axis_parent: np.ndarray = None,
        initial_axis_child: np.ndarray = None,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        subsample_rate: int = 1
    ) -> Dict[str, Union[np.ndarray, bool]]:
        r"""
        Estimates the two joint axes for a biaxial (2-DoF) joint, where each
        axis is constant in its respective local segment frame.

        This algorithm correctly models a biological joint where the axes are
        fixed to the body segments. The core principle is that the relative
        angular velocity, when expressed in the world frame, must lie in the
        plane defined by the two joint axes, which are also expressed in the
        world frame.

        The unknowns being solved for are the constant representations of the
        axes in their local frames:
        - j1_parent: The first axis, constant in the 'self' (parent) frame.
        - j2_child: The second axis, constant in the 'other' (child) frame.

        The cost function at each timestep 't' is:
        e(t) = (w_child_w(t) - w_parent_w(t)) . (j1_w(t) x j2_w(t))
        where:
        - j1_w(t) = R_parent(t) @ j1_parent
        - j2_w(t) = R_child(t) @ j2_child

        Args:
            other (PlateTrial): The PlateTrial for the 'child' segment. 'self'
                is treated as the 'parent' segment.
            initial_axis_parent (np.ndarray, optional): A (3,) initial guess for
                the first joint axis in the PARENT's local frame.
            initial_axis_child (np.ndarray, optional): A (3,) initial guess for
                the second joint axis in the CHILD's local frame.
            max_iterations (int, optional): Max iterations for the optimization.
            tolerance (float, optional): Convergence tolerance.
            subsample_rate (int, optional): Rate to subsample data.

        Returns:
            Dict[str, Union[np.ndarray, bool]]: A dictionary containing:
                - 'axis_parent_local': The optimized (3,) unit axis vector in the parent frame.
                - 'axis_child_local': The optimized (3,) unit axis vector in the child frame.
                - 'converged': A boolean flag indicating if convergence was reached.
        """
        # 1. --- Input Validation and Data Preparation ---
        np.testing.assert_array_almost_equal(
            self.imu_trace.timestamps, other.imu_trace.timestamps, decimal=5,
            err_msg="PlateTrial traces must be time-synchronized."
        )
        indices = np.arange(0, len(self), subsample_rate)

        # 2. --- Transform Gyro Data and Get Rotations ---
        p_imu_trace = self.get_imu_trace_in_global_frame()
        g_p_world = np.array(p_imu_trace.gyro)[indices]
        R_wp = np.array(self.world_trace.rotations)[indices]

        c_imu_trace = other.get_imu_trace_in_global_frame()
        g_c_world = np.array(c_imu_trace.gyro)[indices]
        R_wc = np.array(other.world_trace.rotations)[indices]

        w_rel_world = g_c_world - g_p_world

        # 3. --- Helper Functions for Parametrization ---
        def spherical_to_cartesian(phi, theta):
            return np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)])

        def cartesian_derivatives(phi, theta):
            return (np.array([-np.sin(phi)*np.cos(theta), -np.sin(phi)*np.sin(theta), np.cos(phi)]),
                    np.array([-np.cos(phi)*np.sin(theta), np.cos(phi)*np.cos(theta), 0]))

        def cartesian_to_spherical(axis: np.ndarray):
            phi = np.arcsin(np.clip(axis[2], -1.0, 1.0))
            theta = np.arctan2(axis[1], axis[0])
            return phi, theta

        # 4. --- Initialization of State Vector ---
        # State vector x = [phi1, theta1, phi2, theta2] for j1_parent and j2_child
        if initial_axis_parent is None:
            phi1, theta1 = np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-np.pi, np.pi)
        else:
            phi1, theta1 = cartesian_to_spherical(initial_axis_parent/np.linalg.norm(initial_axis_parent))

        if initial_axis_child is None:
            phi2, theta2 = np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-np.pi, np.pi)
        else:
            phi2, theta2 = cartesian_to_spherical(initial_axis_child/np.linalg.norm(initial_axis_child))

        x = np.array([phi1, theta1, phi2, theta2])

        # 5. --- Gauss-Newton Optimization Loop ---
        for _ in range(max_iterations):
            phi1, theta1, phi2, theta2 = x
            j1_p = spherical_to_cartesian(phi1, theta1)
            j2_c = spherical_to_cartesian(phi2, theta2)
            dj1p_dphi1, dj1p_dtheta1 = cartesian_derivatives(phi1, theta1)
            dj2c_dphi2, dj2c_dtheta2 = cartesian_derivatives(phi2, theta2)

            # Transform local axes to world frame FOR EACH TIMESTEP
            j1_w_t = np.einsum('nij,j->ni', R_wp, j1_p)
            j2_w_t = np.einsum('nij,j->ni', R_wc, j2_c)

            jn_w_t = np.cross(j1_w_t, j2_w_t)
            norm_jn_t = np.linalg.norm(jn_w_t, axis=1, keepdims=True)

            # Avoid division by zero
            if np.any(norm_jn_t < 1e-9):
                print("Warning: Joint axes became parallel during iteration. Stopping.")
                break
            
            jn_w_norm_t = jn_w_t / norm_jn_t
            e = np.einsum('ni,ni->n', w_rel_world, jn_w_norm_t) # Dot product for each row

            # Jacobian 'J' (N, 4) using the chain rule
            dj1w_dphi1 = np.einsum('nij,j->ni', R_wp, dj1p_dphi1)
            dj1w_dtheta1 = np.einsum('nij,j->ni', R_wp, dj1p_dtheta1)
            dj2w_dphi2 = np.einsum('nij,j->ni', R_wc, dj2c_dphi2)
            dj2w_dtheta2 = np.einsum('nij,j->ni', R_wc, dj2c_dtheta2)

            # Derivatives of the plane normal (jn_w)
            djn_dphi1 = np.cross(dj1w_dphi1, j2_w_t)
            djn_dtheta1 = np.cross(dj1w_dtheta1, j2_w_t)
            djn_dphi2 = np.cross(j1_w_t, dj2w_dphi2)
            djn_dtheta2 = np.cross(j1_w_t, dj2w_dtheta2)
            
            # Final Jacobian columns
            J0 = np.einsum('ni,ni->n', w_rel_world, djn_dphi1 / norm_jn_t)
            J1 = np.einsum('ni,ni->n', w_rel_world, djn_dtheta1 / norm_jn_t)
            J2 = np.einsum('ni,ni->n', w_rel_world, djn_dphi2 / norm_jn_t)
            J3 = np.einsum('ni,ni->n', w_rel_world, djn_dtheta2 / norm_jn_t)

            J = np.column_stack((J0, J1, J2, J3))
            
            try:
                delta_x = -np.linalg.pinv(J) @ e
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in pseudoinverse. Stopping.")
                break

            x += delta_x
            if np.linalg.norm(delta_x) < tolerance:
                j1_res = spherical_to_cartesian(x[0], x[1])
                j2_res = spherical_to_cartesian(x[2], x[3])
                return {'axis_parent_local': j1_res, 'axis_child_local': j2_res, 'converged': True}

        # 6. --- Return final (non-converged) estimate ---
        j1_res = spherical_to_cartesian(x[0], x[1])
        j2_res = spherical_to_cartesian(x[2], x[3])
        return {'axis_parent_local': j1_res, 'axis_child_local': j2_res, 'converged': False}

    def generate_1dof_plate(
        self,
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

        parent_world_trace = self.world_trace
        num_samples = len(parent_world_trace)
        duration = parent_world_trace.timestamps[-1] - parent_world_trace.timestamps[0]

        angle = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi)
        R_joint_motion = Rotation.from_euler('z', angle)

        R_parent_matrices = np.stack(self.world_trace.rotations)
        R_p2j_mat = parent_to_joint_rotation.as_matrix()
        R_c2j_inv_mat = child_to_joint_rotation.as_matrix().T
        R_rel_total_matrices = R_p2j_mat @ R_joint_motion.as_matrix() @ R_c2j_inv_mat
        R_child_matrices = R_parent_matrices @ R_rel_total_matrices

        P_parent = np.stack(self.world_trace.positions)
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

        return PlateTrial(f"{self.name}_child_dof1", child_imu_trace, child_world_trace)


    def generate_2dof_plate(
        self,
        joint_center_parent: np.ndarray,
        joint_center_child: np.ndarray,
        parent_to_joint_rotation: Rotation = None,
        child_to_joint_rotation: Rotation = None,
        add_noise: bool = True,
        gyro_noise_std: float = 0.005,
        acc_noise_std: float = 0.05
    ) -> 'PlateTrial':
        """
        Generates a child PlateTrial connected by a 2-DOF universal joint.
        The kinematic chain is:
        R_child = R_parent @ R_p2j @ R_c2j.inv()
        """
        if parent_to_joint_rotation is None: parent_to_joint_rotation = Rotation.random()
        if child_to_joint_rotation is None: child_to_joint_rotation = Rotation.random()

        parent_world_trace = self.world_trace
        num_samples = len(parent_world_trace)
        duration = parent_world_trace.timestamps[-1] - parent_world_trace.timestamps[0]

        p_angles = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi/2)
        c_angles = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi/2)

        R_parent_matrices = np.stack(self.world_trace.rotations)
        parent_axis = parent_to_joint_rotation.as_rotvec() / np.linalg.norm(parent_to_joint_rotation.as_rotvec()) if parent_to_joint_rotation.as_rotvec().any() != 0 else np.array([1.0, 0.0, 0.0])
        normalized_axis_stacked = np.tile(parent_axis, (num_samples, 1))
        scaled_rotation_vectors = normalized_axis_stacked * p_angles[:, np.newaxis]
        R_p2j_mat = Rotation.from_rotvec(scaled_rotation_vectors).as_matrix()

        child_axis = child_to_joint_rotation.as_rotvec() / np.linalg.norm(child_to_joint_rotation.as_rotvec()) if child_to_joint_rotation.as_rotvec().any() != 0 else np.array([1.0, 0.0, 0.0])
        normalized_axis_stacked_c = np.tile(child_axis, (num_samples, 1))
        scaled_rotation_vectors_c = normalized_axis_stacked_c * c_angles[:, np.newaxis]
        R_c2j_inv_mat = Rotation.from_rotvec(scaled_rotation_vectors_c).inv().as_matrix()
        R_rel_total_matrices = R_p2j_mat @ R_c2j_inv_mat
        R_child_matrices = R_parent_matrices @ R_rel_total_matrices

        P_parent = np.stack(self.world_trace.positions)
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

        return PlateTrial(f"{self.name}_child_dof2", child_imu_trace, child_world_trace)


    def generate_3dof_plate(
        self,
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

        parent_world_trace = self.world_trace
        num_samples = len(parent_world_trace)
        duration = parent_world_trace.timestamps[-1] - parent_world_trace.timestamps[0]

        angles = [_generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi/2) for _ in range(3)]
        R_joint_motion = Rotation.from_euler('zyx', np.vstack(angles).T)

        R_parent_matrices = np.stack(self.world_trace.rotations)
        R_p2j_mat = parent_to_joint_rotation.as_matrix()
        R_c2j_inv_mat = child_to_joint_rotation.as_matrix().T
        R_rel_total_matrices = R_p2j_mat @ R_joint_motion.as_matrix() @ R_c2j_inv_mat
        R_child_matrices = R_parent_matrices @ R_rel_total_matrices

        P_parent = np.stack(self.world_trace.positions)
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

        return PlateTrial(f"{self.name}_child_dof3", child_imu_trace, child_world_trace)