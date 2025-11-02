import os
import numpy as np
import scipy.signal as signal
from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace
from typing import Tuple, List, Dict
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

    def find_2dof_joint_axes_from_relative_orientation(self,
                                                        other_plate: 'PlateTrial',
                                                        verbose: bool = False 
                                                        ) -> dict:
        """
        Estimates the two axes of a 2-DoF joint from relative orientation.

        This method implements a "carrying angle" optimization. It assumes that
        the relative motion between this plate (segment 1) and the `other_plate`
        (segment 2) can be described by a 2-DoF joint (e.g., a knee).

        The core principle is that there exist "ideal" coordinate frames for
        segment 1 and segment 2, related to their original frames by static
        rotations (R1 and R2), such that the relative motion between them,
        when expressed as a Z-X'-Y" Euler sequence, has a *constant* X' angle
        (the "carrying angle").

        The function optimizes for the static rotations R1 and R2 that minimize
        the variance of this carrying angle over the entire motion. From R1 and
        R2, it computes the joint axes (j1 and j2) in the original segment frames.

        Args:
            other_plate (PlateTrial): The PlateTrial object for the *other*
                body segment (segment 2).
            verbose (bool, optional): If True, prints the `least_squares`
                optimizer's termination report. Defaults to False.

        Returns:
            dict: A dictionary containing:
                - 'axis_in_frame1': (3,) ndarray. The estimated unit axis vector
                  (j1) for the first rotation, in segment 1's original frame.
                - 'axis_in_frame2': (3,) ndarray. The estimated unit axis vector
                  (j2) for the second rotation, in segment 2's original frame.
                - 'carrying_angle_rad': The estimated constant carrying angle in
                  radians.
                - 'success': bool. True if the optimizer converged.
                - 'message': str. The convergence message from the optimizer.
        """
        # --- 1. Input Validation ---
        assert isinstance(other_plate, self.__class__), "other_plate must be a PlateTrial instance"
        assert len(self) == len(other_plate), "PlateTrials must have the same length"
        
        # --- 2. Get Relative Rotations ---
        # Get rotations from world frame to segment 1 (self) and segment 2 (other)
        self_rotations = Rotation.from_matrix(self.world_trace.rotations)
        other_rotations = Rotation.from_matrix(other_plate.world_trace.rotations)

        # R_rel = R_seg1_to_seg2 = R_world_to_seg1.inv() * R_world_to_seg2
        R_rel = (self_rotations.inv() * other_rotations)
        
        # --- 3. Define Helper Functions for Carrying Angle ---
        
        def _get_carrying_angle_from_quat(q: np.ndarray) -> float:
            """
            Calculates the second Euler angle (beta) from a z-x'-y" sequence
            for a *single* quaternion.
            
            q = [w, x, y, z]
            beta = arcsin(2 * (w*x + y*z))
            """
            qw, qx, qy, qz = q
            # Clip to avoid floating point errors with arcsin
            val = np.clip(2 * (qw * qx + qy * qz), -1.0, 1.0)
            return np.arcsin(val)

        def _get_carrying_angles_vec(q: np.ndarray) -> np.ndarray:
            """
            Vectorized version of _get_carrying_angle_from_quat for an
            (N, 4) array of quaternions.
            
            q = [w, x, y, z]
            beta = arcsin(2 * (w*x + y*z))
            """
            qw = q[:, 0]
            qx = q[:, 1]
            qy = q[:, 2]
            qz = q[:, 3]
            val = np.clip(2 * (qw * qx + qy * qz), -1.0, 1.0)
            return np.arcsin(val)

        # --- 4. Define The Cost (Residuals) Function for Optimization ---
        
        def _residuals(params: np.ndarray) -> np.ndarray:
            """
            The cost function to be minimized by `least_squares`.
            
            It calculates the deviation of the carrying angle from its mean
            for the entire motion. Minimizing this minimizes the variance.

            Args:
                params (np.ndarray): A (6,) array [rx1, ry1, rz1, rx2, ry2, rz2]
                    containing the rotation vectors for R1 and R2.
            """
            # Unpack the static alignment rotations R1 and R2
            rot_vec1 = params[0:3]
            rot_vec2 = params[3:6]
            R1 = Rotation.from_rotvec(rot_vec1)
            R2 = Rotation.from_rotvec(rot_vec2)

            # Apply alignment rotations to get the "calibrated" relative rotation
            # R_calibrated = R1 * R_relative * R2_inverse
            R_calibrated = R1 * R_rel * R2.inv()

            # Convert to quaternions for Euler angle extraction
            # .as_quat() returns [x, y, z, w]
            q_cal_w_last = R_calibrated.as_quat() 
            # Helper function expects [w, x, y, z]
            q_cal_w_first = q_cal_w_last[:, [3, 0, 1, 2]] 

            # Calculate the carrying angle for every timestep
            carrying_angles = _get_carrying_angles_vec(q_cal_w_first)
            
            # The error is the deviation of each angle from the mean angle.
            # Minimizing the sum of squares of this residual minimizes the variance.
            return carrying_angles - np.mean(carrying_angles)

        # --- 5. Run the Optimization ---
        
        # Initial guess: No rotation needed (R1 and R2 are identity)
        initial_params = np.zeros(6)

        # Run the non-linear least squares optimization
        result = least_squares(
            _residuals, 
            initial_params, 
            method='trf', 
            ftol=1e-6,
            gtol=1e-6,
            verbose=(2 if verbose else 0) # Use level 2 for full report
        )

        # --- 6. Extract and Compute Final Results ---
        
        # Get the optimized rotation vectors and create Rotation objects
        optimal_params = result.x
        R1_opt = Rotation.from_rotvec(optimal_params[0:3])
        R2_opt = Rotation.from_rotvec(optimal_params[3:6])

        # The "ideal" axes in the calibrated frames are z and y
        # j1 (1st rotation) is z-axis, j2 (3rd rotation) is y-axis
        j1_ideal = np.random.normal(size=3)
        j1_ideal /= np.linalg.norm(j1_ideal)
        j2_ideal = np.random.normal(size=3)
        j2_ideal /= np.linalg.norm(j2_ideal)

        # Transform these ideal axes back to the original segment frames
        # j_original_1 = R1.inv() * j_ideal_1
        # j_original_2 = R2.inv() * j_ideal_2
        axis_in_frame1 = R1_opt.apply(j1_ideal, inverse=True)
        axis_in_frame2 = R2_opt.apply(j2_ideal, inverse=True)
        
        # Recalculate the final mean carrying angle
        # We can find the mean by taking the carrying angle at the first
        # timestep and subtracting its residual (which is C[0] - mean(C)).
        # C[0] - (C[0] - mean(C)) = mean(C)
        final_residuals = _residuals(optimal_params)
        
        # Get calibrated rotation at first timestep
        R_cal_first = R1_opt * R_rel[0] * R2_opt.inv()
        q_cal_first_w_last = R_cal_first.as_quat() # [x, y, z, w]
        q_cal_first_w_first = q_cal_first_w_last[[3, 0, 1, 2]] # [w, x, y, z]
        
        # Use the scalar helper for this single calculation
        first_carrying_angle = _get_carrying_angle_from_quat(q_cal_first_w_first)
        final_mean_carrying_angle = first_carrying_angle - final_residuals[0]

        return {
            'axis_in_frame1': axis_in_frame1,
            'axis_in_frame2': axis_in_frame2,
            'carrying_angle_rad': final_mean_carrying_angle,
            'success': result.success,
            'message': result.message
        }