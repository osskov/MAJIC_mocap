import os
import numpy as np
import scipy.signal as signal
from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

IMU_TO_TRC_NAME_MAP = {
    'pelvis_imu': 'Pelvis_IMU', 'femur_r_imu': 'R.Femur_IMU', 'femur_l_imu': 'L.Femur_IMU',
    'tibia_r_imu': 'R.Tibia_IMU', 'tibia_l_imu': 'L.Tibia_IMU', 'calcn_r_imu': 'R.Foot_IMU',
    'calcn_l_imu': 'L.Foot_IMU', 'torso_imu': 'Back_IMU'
}

class PlateTrial:
    """
    This class contains a trace of a plate frame over time.
    """

    name: str
    imu_trace: IMUTrace
    world_trace: WorldTrace

    def __init__(self, name: str, imu_trace: IMUTrace, world_trace: WorldTrace):
        assert len(imu_trace) == len(world_trace), "IMU and World traces must have the same length"
        if max(np.abs(imu_trace.timestamps - world_trace.timestamps)) > 1e-8:
            print(f"IMU and World traces must have the same timestamps. Max difference: {max(np.abs(imu_trace.timestamps - world_trace.timestamps))}")
        assert max(np.abs(imu_trace.timestamps - world_trace.timestamps)) < 1e-8, "Timestamps must match"
        assert isinstance(imu_trace, IMUTrace)
        assert isinstance(world_trace, WorldTrace)

        self.name = name
        self.imu_trace = imu_trace
        self.world_trace = world_trace


    def __len__(self):
        return len(self.imu_trace)

    def __getitem__(self, key):
        return PlateTrial(self.name, self.imu_trace[key], self.world_trace[key])
    
    def copy(self) -> 'PlateTrial':
        """
        Returns a deep copy of the PlateTrial object.

        Note: This relies on IMUTrace and WorldTrace having a functioning .copy() method
                to ensure the underlying data arrays are also duplicated.
        """
        return PlateTrial(
            self.name,
            self.imu_trace.copy(),
            self.world_trace.copy()
        )

    def _align_world_trace_to_imu_trace(self) -> 'PlateTrial':
        synthetic_imu_trace = self.world_trace.calculate_imu_trace(skip_lin_acc=True)
        R_wt_it = synthetic_imu_trace.calculate_rotation_offset_from_gyros(self.imu_trace)
        new_world_rotations = [rot @ R_wt_it for rot in self.world_trace.rotations]
        new_world_trace = WorldTrace(self.world_trace.timestamps, self.world_trace.positions, new_world_rotations)
        return PlateTrial(self.name, self.imu_trace, new_world_trace)

    def project_imu_trace(self, local_offset: np.ndarray) -> IMUTrace:
        """
        This function estimates the values for an IMUTrace at a different location relative to the plate.
        """
        return self.imu_trace.project_acc(local_offset)
    
    @staticmethod
    def generate_plate_from_traces(
        imu_traces: Dict[str, 'IMUTrace'], 
        world_traces: Dict[str, 'WorldTrace'], 
        align_plate_trials: bool
    ) -> List['PlateTrial']:
        """
        Private helper to synchronize, align, and trim IMU and World traces.
        """
        plate_trials = []
        imu_slice, world_slice = slice(0, 0), slice(0, 0)
        
        if not imu_traces:
            print("Warning: No IMU traces loaded.")
            return []
        
        for imu_name, imu_trace in imu_traces.items():
            try:
                world_trace = world_traces[imu_name] if imu_name in world_traces else world_traces[IMU_TO_TRC_NAME_MAP[imu_name]]
            except KeyError:
                print(f"IMU {imu_name} not found in world traces. Skipping.")
                continue

            # Resample if frequencies don't match
            if abs(imu_trace.get_sample_frequency() - world_trace.get_sample_frequency()) > 0.2:
                # print(f"Sample frequency mismatch for {imu_name}: IMU {imu_trace.get_sample_frequency()} Hz, World {world_trace.get_sample_frequency()} Hz")
                imu_trace = imu_trace.resample(float(world_trace.get_sample_frequency()))

            # Sync traces and create PlateTrial
            imu_slice, world_slice = PlateTrial._sync_traces(imu_trace, world_trace)
            synced_imu_trace = imu_trace[imu_slice].re_zero_timestamps()
            synced_world_trace = world_trace[world_slice].re_zero_timestamps()
            
            new_plate_trial = PlateTrial(imu_name, synced_imu_trace, synced_world_trace)
            
            if align_plate_trials:
                new_plate_trial = new_plate_trial._align_world_trace_to_imu_trace()
                
            plate_trials.append(new_plate_trial)

        # Trim all trials to the same minimum length
        plate_trial_lengths = [len(plate_trial) for plate_trial in plate_trials]
        if len(set(plate_trial_lengths)) > 1:
            print(f"Warning: Plate trials have different lengths: {plate_trial_lengths}, Trimming to min length.")
            min_length = min(plate_trial_lengths)
            plate_trials = [plate_trial[:min_length] for plate_trial in plate_trials]

        return plate_trials

    @staticmethod
    def load_trial_from_Al_Borno_folder(folder_path: str, align_plate_trials=True) -> List['PlateTrial']:
        """
        Loads a trial from the Al Borno data structure.
        - IMU data in /IMU
        - Mocap data in /Mocap
        """
        # 1. Load IMU Traces
        imu_folder = os.path.join(folder_path, 'IMU')
        imu_traces = IMUTrace.load_IMUTraces_from_Al_Borno_folder(imu_folder)

        # 2. Load World Traces
        mocap_folder = os.path.join(folder_path, 'Mocap/')
        trc_file_path = [file for file in os.listdir(mocap_folder) if file.endswith('.trc') and 'static' not in file][0]
        trc_file_path = os.path.abspath(os.path.join(mocap_folder, trc_file_path))
        world_traces = WorldTrace.load_from_trc_file(trc_file_path)

        print(f"Loaded {len(imu_traces)} IMU traces and {len(world_traces)} World traces. Generating PlateTrials...")

        # 3. Process all traces
        plate_trials = PlateTrial.generate_plate_from_traces(
            imu_traces, world_traces, align_plate_trials
        )

        return plate_trials
    
    @staticmethod
    def load_trial_from_folder(folder_path: str, align_plate_trials=True) -> List['PlateTrial']:
        """
        Loads a trial from the Skov data structure.
        - IMU data in /imu data
        - Mocap data in root folder
        """
        # 1. Load IMU Traces
        imu_traces = IMUTrace.load_IMUTraces_from_Skov_folder(folder_path)

        # 2. Load World Traces
        trc_file_path = [file for file in os.listdir(folder_path) if file.endswith('.trc')][0]
        trc_file_path = os.path.abspath(os.path.join(folder_path, trc_file_path))
        world_traces = WorldTrace.load_from_trc_file(trc_file_path)

        print(f"Loaded {len(imu_traces)} IMU traces and {len(world_traces)} World traces. Generating PlateTrials...")

        # 3. Process all traces
        plate_trials = PlateTrial.generate_plate_from_traces(
            imu_traces, world_traces, align_plate_trials
        )
        
        return plate_trials
    
    @staticmethod
    def _sync_traces(imu_trace: IMUTrace, world_trace: WorldTrace) -> Tuple[slice, slice]:
        if not np.isclose(imu_trace.get_sample_frequency(), world_trace.get_sample_frequency(), rtol=0.2):
            imu_trace = imu_trace.resample(float(world_trace.get_sample_frequency()))

        synthetic_imu_trace = world_trace.calculate_imu_trace(skip_lin_acc=True)
        imu_slice, world_slice = PlateTrial._sync_arrays(
            np.linalg.norm(imu_trace.gyro, axis=1),
            np.linalg.norm(synthetic_imu_trace.gyro, axis=1)
        )
        return imu_slice, world_slice

    @staticmethod
    def _sync_arrays(array1: np.ndarray, array2: np.ndarray) -> Tuple[slice, slice]:
        assert array1.ndim == array2.ndim == 1
        max_len = max(len(array1), len(array2))
        a1 = np.pad(array1, (0, max_len - len(array1)), mode='constant')
        a2 = np.pad(array2, (0, max_len - len(array2)), mode='constant')
        lag = np.argmax(signal.correlate(a1, a2, mode='full')) - (max_len - 1)
        i1, i2 = max(0, lag), max(0, -lag)
        new_len = min(len(array1) - i1, len(array2) - i2)
        return slice(i1, i1 + new_len), slice(i2, i2 + new_len)
    
    def get_imu_trace_in_global_frame(self) -> IMUTrace:
        """
        Rotates the IMU trace data into the global frame using the world trace rotations.
        """
        rotated_acc = [r @ a for r, a in zip(self.world_trace.rotations, self.imu_trace.acc)]
        rotated_gyro = [r @ g for r, g in zip(self.world_trace.rotations, self.imu_trace.gyro)]
        rotated_mag = [r @ m for r, m in zip(self.world_trace.rotations, self.imu_trace.mag)]
        
        return IMUTrace(
            timestamps=self.imu_trace.timestamps,
            acc=rotated_acc,
            gyro=rotated_gyro,
            mag=rotated_mag)

    def find_2dof_joint_axes_from_relative_orientation(self,
        other_plate: 'PlateTrial'
    ) -> dict:
        """
        Estimates the two axes of a 2-DoF joint from a time history of relative orientation,
        using scipy.spatial.transform.Rotation for quaternion operations.

        This method is based on the principle that a 2-DoF joint motion can be
        described by a sequence of rotations (e.g., z-x'-y" Euler) where the middle
        rotation angle (the 'carrying angle') remains constant.

        The function optimizes for the two static rotations that transform the initial
        body segment coordinate frames into new, "ideal" frames where the variance of
        the carrying angle is minimized. From these static rotations, it computes
        the joint axes in the original frames.

        Args:
            relative_orientations (list[np.ndarray]): A list of quaternions ([w, x, y, z])
                representing the orientation of body segment 2 relative to body
                segment 1 for each time step.

        Returns:
            dict: A dictionary containing:
                - 'axis_in_frame1': The estimated 3D unit axis vector (j1) for the first
                rotation, expressed in the original coordinate frame of body segment 1.
                - 'axis_in_frame2': The estimated 3D unit axis vector (j2) for the second
                rotation, expressed in the original coordinate frame of body segment 2.
                - 'carrying_angle_rad': The estimated constant carrying angle in radians.
                - 'success': A boolean from the optimizer indicating if it converged.
                - 'message': The convergence message from the optimizer.
        """
        assert other_plate is not None, "Other PlateTrial must be provided"
        assert isinstance(other_plate, PlateTrial), "other_plate must be a PlateTrial instance"
        assert len(self) == len(other_plate), "PlateTrials must have the same length"
        self_rotations = Rotation.from_matrix(self.world_trace.rotations)
        other_rotations = Rotation.from_matrix(other_plate.world_trace.rotations)

        R_rel = (self_rotations.inv() * other_rotations)
        
        def _get_carrying_angle_from_quat(q: np.ndarray) -> float:
            """
            Calculates the second Euler angle (beta) from a z-x'-y" sequence,
            based on the explicit formula from Laidig et al. (2022), Eq. 23.

            Args:
                q (np.ndarray): A single quaternion in [w, x, y, z] format.

            Returns:
                float: The carrying angle in radians.
            """
            qw, qx, qy, qz = q
            # Ensure the argument for arcsin is within [-1, 1] to avoid NaN errors
            val = np.clip(2 * (qw * qx + qy * qz), -1.0, 1.0)
            return np.arcsin(val)
    
        def _residuals(params):
            """
            The cost function to be minimized. It calculates the deviation of the
            carrying angle from its mean for the entire motion.
            """
            # Parameters are two 3D rotation vectors for the frame alignments
            rot_vec1 = params[0:3]
            rot_vec2 = params[3:6]

            # Convert rotation vectors to Rotation objects
            R1 = Rotation.from_rotvec(rot_vec1)
            R2 = Rotation.from_rotvec(rot_vec2)

            # Apply alignment rotations using efficient, vectorized multiplication:
            # R_calibrated = R1 * R_relative * R2_inverse
            R_calibrated = R1 * R_rel * R2.inv()

            # Convert the stack of calibrated rotations back to [w, x, y, z] quaternions
            q_cal_w_last = R_calibrated.as_quat()
            q_cal_w_first = q_cal_w_last[:, [3, 0, 1, 2]]

            # Calculate the carrying angle for each time step
            carrying_angles = np.array([_get_carrying_angle_from_quat(q) for q in q_cal_w_first])
            
            # The error is the deviation from the mean, which forces the variance to be small
            return carrying_angles - np.mean(carrying_angles)

        # Initial guess: No rotation needed for either frame
        initial_params = np.zeros(6)

        # Run the non-linear least squares optimization
        result = least_squares(_residuals, initial_params, method='trf', ftol=1e-6)

        # --- Extract and compute final results from the optimal parameters ---
        optimal_params = result.x
        R1_opt = Rotation.from_rotvec(optimal_params[0:3])
        R2_opt = Rotation.from_rotvec(optimal_params[3:6])

        # In the "ideal" calibrated frame, the joint axes are standard basis vectors
        j1_ideal = np.array([0, 0, 1])  # z-axis for flexion/extension
        j2_ideal = np.array([0, 1, 0])  # y-axis for pronation/supination

        # To find the axes in the original frames, we apply the inverse of the
        # optimized alignment rotations.
        axis_in_frame1 = R1_opt.apply(j1_ideal, inverse=True)
        axis_in_frame2 = R2_opt.apply(j2_ideal, inverse=True)
        
        # Recalculate the final mean carrying angle from the optimized alignment
        final_residuals = _residuals(optimal_params)
        
        # carrying_angles = final_residuals + mean_angle.
        # So, mean_angle = first_carrying_angle - first_residual.
        R_cal_first = R1_opt * R_rel[0] * R2_opt.inv()
        q_cal_first_w_last = R_cal_first.as_quat()
        q_cal_first_w_first = q_cal_first_w_last[[3, 0, 1, 2]]
        
        final_mean_carrying_angle = _get_carrying_angle_from_quat(q_cal_first_w_first) - final_residuals[0]

        return {
            'axis_in_frame1': axis_in_frame1,
            'axis_in_frame2': axis_in_frame2,
            'carrying_angle_rad': final_mean_carrying_angle,
            'success': result.success,
            'message': result.message
        }