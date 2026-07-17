import os
from .IMUTrace import IMUTrace
from typing import List, Tuple, Union
import numpy as np
from .finite_difference_utils import central_difference
from .gyro_utils import finite_difference_rotations
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from typing import Dict
import xml.etree.ElementTree as ET
import pandas as pd

def _generate_smooth_motion_profile(
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

class WorldTrace:
    """
    This class contains a trace of a world frame over time. Optionally, this can attach an IMUTrace and manipulate it.
    Or, it can generate a synthetic trace by finite differencing the world frames over time.
    """

    def __init__(self, timestamps: np.ndarray, positions: Union[List[np.ndarray], np.ndarray], rotations: Union[List[np.ndarray], np.ndarray]):
        self.timestamps = timestamps
        self.positions = np.asarray(positions)
        self.rotations = np.asarray(rotations)

    def __len__(self):
        """
        Returns the number of samples in the WorldTrace. This allows us to call len(trace) on a WorldTrace instance.
        """
        return len(self.timestamps)

    def __sub__(self, other: 'WorldTrace') -> 'WorldTrace':
        """
        Allows us to subtract two WorldTrace instances. This will subtract the positions and rotations of the two traces.
        """
        if len(self) != len(other):
            raise ValueError(f"WorldTraces must have the same length to subtract them. Got self {len(self)} and other {len(other)}.")
        assert np.array_equal(self.timestamps[0],
                              other.timestamps[0]), "WorldTraces must have the same start time to subtract them."
        return WorldTrace(self.timestamps, self.positions - other.positions,
                          np.matmul(self.rotations, other.rotations.transpose(0, 2, 1)))

    def __getitem__(self, key) -> 'WorldTrace':
        """
        Allows us to use the square bracket notation to access the WorldTrace instance. This allows us to slice the
        WorldTrace instance and access ranges of items with `sub_trace = trace[1:4]`. If we pass an integer, we can
        return the corresponding item as a length 1 trace with `sub_trace = trace[2]` and `len(sub_trace) == 1`.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new WorldTrace instance with the sliced items
            return WorldTrace(self.timestamps[key], self.positions[key], self.rotations[key])
        else:
            # If key is an integer, return the corresponding item as a length 1 trace
            return WorldTrace(np.array([self.timestamps[key]]), self.positions[key:key+1], self.rotations[key:key+1])

    def __eq__(self, other):
        """
        Allows us to compare two WorldTrace instances for equality. This will return True if the timestamps, positions, and
        rotations are all _exactly_ equal.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions.shape[0] != other.positions.shape[0] or self.rotations.shape[0] != other.rotations.shape[0]:
            return False
        return (np.array_equal(self.timestamps, other.timestamps) and
                np.array_equal(self.positions, other.positions) and
                np.array_equal(self.rotations, other.rotations))

    def transform(self, rotate: np.ndarray = np.eye(3), translate: np.ndarray = np.zeros(3)) -> 'WorldTrace':
        """
        This function transforms the WorldTrace by rotating and translating the positions and rotations.
        """
        return WorldTrace(self.timestamps,
                          np.einsum('ij,nj->ni', rotate, self.positions) + translate,
                          np.matmul(rotate, self.rotations))

    def allclose(self, other, atol=1e-6):
        """
        Allows us to compare two WorldTrace instances for approximate equality. This will return True if the timestamps,
        positions, and rotations are all approximately equal within the specified tolerance.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions.shape[0] != other.positions.shape[0] or self.rotations.shape[0] != other.rotations.shape[0]:
            return False
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                np.allclose(self.positions, other.positions, atol=atol) and
                np.allclose(self.rotations, other.rotations, atol=atol))
    
    def copy(self) -> 'WorldTrace':
        """
        Returns a deep copy of the WorldTrace object.
        """
        return WorldTrace(
            self.timestamps.copy(),
            self.positions.copy(),
            self.rotations.copy()
        )
    
    def resample(self, new_frequency: float) -> 'WorldTrace':
        """
        This function resamples the WorldTrace to a new, specified frequency using linear interpolation for
        positions and spherical linear interpolation (slerp) for rotations.

        Args:
            new_frequency: The desired sampling frequency in Hz.

        Returns:
            A new WorldTrace instance resampled to the new frequency.
        """
        if new_frequency <= 0:
            raise ValueError("New frequency must be a positive number.")

        original_timestamps = self.timestamps
        if len(original_timestamps) < 2:
            return self.copy() # Cannot resample a trace with 0 or 1 samples

        # 1. Determine the new timestamps, prioritizing a perfect time step (dt)
        start_time = original_timestamps[0]
        last_original_time = original_timestamps[-1]
        new_dt = 1.0 / new_frequency
        
        # Define a small tolerance (epsilon) based on the new time step. 
        # This is used to make the 'stop' value in np.arange inclusive.
        # We use a fraction of the new_dt to ensure precision.
        epsilon = new_dt * 1e-6 
        
        # Calculate the theoretical end of the uniform grid. 
        # This is the last point generated by the perfect steps.
        time_duration = last_original_time - start_time
        num_intervals = np.round(time_duration / new_dt)
        theoretical_last_step = start_time + num_intervals * new_dt
        
        # np.arange creates a perfectly uniform time series
        new_timestamps = np.arange(start=start_time, 
                                   stop=theoretical_last_step, 
                                   step=new_dt)

        # 2. Interpolate Positions (Linear Interpolation)
        # Create interpolation functions for each dimension (x, y, z)
        interp_x = np.interp(new_timestamps, original_timestamps, self.positions[:, 0])
        interp_y = np.interp(new_timestamps, original_timestamps, self.positions[:, 1])
        interp_z = np.interp(new_timestamps, original_timestamps, self.positions[:, 2])

        # Combine interpolated axes back into a single array
        new_positions = np.column_stack((interp_x, interp_y, interp_z))

        # 3. Interpolate Rotations (SLERP via Quaternions)
        # Convert 3x3 rotation matrices to quaternions (x, y, z, w)
        original_quats = Rotation.from_matrix(self.rotations).as_quat(canonical=True)

        # Create a Rotation object for interpolation
        # from_quat creates a set of rotations
        original_rotations = Rotation.from_quat(original_quats)

        # Slerp the rotations to the new timestamps
        # The 'Slerp' method internally uses the original timestamps for interpolation
        slerp = Slerp(original_timestamps, original_rotations)
        new_rotations_obj = slerp(new_timestamps)

        # Convert interpolated Rotation objects back to an array of 3x3 matrices
        new_rotations = new_rotations_obj.as_matrix().copy()

        # 4. Return the new WorldTrace
        return WorldTrace(new_timestamps, new_positions, new_rotations)

    def finite_difference_world_frame_accelerations(self, acc_from_gravity: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        This function computes the acceleration of the world frame by finite differencing the positions.
        """
        acc_axis = []
        for axis in range(3):
            vel_axis = central_difference(self.positions[:, axis], self.timestamps)
            acc_axis.append(central_difference(vel_axis, self.timestamps))
        return np.column_stack(acc_axis) + acc_from_gravity

    def calculate_imu_trace(self,
                            acc_from_gravity: np.ndarray = np.zeros(3),
                            magnetic_field: np.ndarray = np.zeros(3),
                            skip_lin_acc=False) -> IMUTrace:
        """
        This function computes the IMU trace from the world trace by finite differencing the positions and rotations.
        """
        rotations_np = np.array(self.rotations)
        if not skip_lin_acc:
            world_acc = self.finite_difference_world_frame_accelerations(acc_from_gravity)
            world_acc_np = np.array(world_acc)
            local_acc = np.einsum('nji,nj->ni', rotations_np, world_acc_np)
        else:
            local_acc = np.einsum('nji,j->ni', rotations_np, acc_from_gravity)
        assert isinstance(magnetic_field, np.ndarray)
        local_mag = np.einsum('nji,j->ni', rotations_np, magnetic_field)
        local_gyros = finite_difference_rotations(self.rotations, self.timestamps)
        return IMUTrace(self.timestamps, local_gyros, local_acc, local_mag)

    def re_zero_timestamps(self) -> 'WorldTrace':
        """
        Start timestamps at 0
        """
        return WorldTrace(self.timestamps - self.timestamps[0], self.positions, self.rotations)


    @staticmethod
    def construct_from_markers(timestamps: np.ndarray, marker_o: np.ndarray, marker_d: np.ndarray, marker_x: np.ndarray,
                               marker_y: np.ndarray):
        """
        This function constructs a WorldTrace from three markers. This is useful for generating synthetic data.
        """

        assert not np.isnan(marker_o).any(), "NaN in marker_o"
        assert not np.isnan(marker_d).any(), "NaN in marker_d"
        assert not np.isnan(marker_x).any(), "NaN in marker_x"
        assert not np.isnan(marker_y).any(), "NaN in marker_y"

        # Constructing axis and orientation components
        x_axis_1 = marker_x - marker_d
        x_axis_1 = x_axis_1 / np.linalg.norm(x_axis_1, axis=1)[:, None]
        assert not np.isnan(x_axis_1).any(), "NaN in x_axis_1"
        x_axis_2 = marker_o - marker_y
        x_axis_2 = x_axis_2 / np.linalg.norm(x_axis_2, axis=1)[:, None]
        if np.isnan(x_axis_2).any():
            x_axis = x_axis_1
        else:
            x_axis = (x_axis_1 + x_axis_2) / 2
        x_axis = x_axis / np.linalg.norm(x_axis, axis=1)[:, None]
        assert not np.isnan(x_axis).any(), "NaN in x_axis"

        y_axis_temp_1 = marker_o - marker_x
        y_axis_temp_1 = y_axis_temp_1 / np.linalg.norm(y_axis_temp_1, axis=1)[:, None]
        assert not np.isnan(y_axis_temp_1).any(), "NaN in y_axis_temp_1"
        y_axis_temp_2 = marker_y - marker_d
        y_axis_temp_2 = y_axis_temp_2 / np.linalg.norm(y_axis_temp_2, axis=1)[:, None]
        assert not np.isnan(y_axis_temp_2).any(), "NaN in y_axis_temp_2"
        y_axis_temp = (y_axis_temp_1 + y_axis_temp_2) / 2
        y_axis_temp = y_axis_temp / np.linalg.norm(y_axis_temp, axis=1)[:, None]
        assert not np.isnan(y_axis_temp).any(), "NaN in y_axis_temp"

        z_axis = np.cross(x_axis, y_axis_temp)
        assert not np.isnan(z_axis).any(), "NaN in z_axis"
        z_axis = z_axis / np.linalg.norm(z_axis, axis=1)[:, None]
        assert not np.isnan(z_axis).any(), "NaN in z_axis"
        y_axis = np.cross(z_axis, x_axis)
        assert not np.isnan(y_axis).any(), "NaN in y_axis"

        error_y = np.linalg.norm(y_axis - y_axis_temp_1, axis=1)
        angle_error = np.arccos(np.clip(np.sum(y_axis * y_axis_temp_1, axis=1), -1, 1)) * 180 / np.pi
        if np.mean(angle_error) > 1.0:
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 5)
            # ax[0].plot(angle_error)
            # ax[0].set_title("Angle Error (deg)")
            # ax[1].plot(np.linalg.norm(marker_o - marker_x, axis=1), label='o-x')
            # ax[1].plot(np.linalg.norm(marker_o - marker_y, axis=1), label='o-y')
            # ax[1].plot(np.linalg.norm(marker_o - marker_d, axis=1), label='o-d')
            # ax[2].plot(np.linalg.norm(marker_x - marker_y, axis=1), label='x-y')
            # ax[2].plot(np.linalg.norm(marker_x - marker_d, axis=1), label='x-d')
            # ax[3].plot(np.linalg.norm(marker_y - marker_d, axis=1), label='y-d')
            # ax[1].set_title("Marker Distances from O")
            # ax[1].legend()
            # ax[2].set_title("Marker Distances from X")
            # ax[2].legend()
            # ax[3].set_title("Marker Distances from Y")
            # ax[3].legend()
            # ax[4].plot(timestamps, error_y, label='y-y_temp')
            # plt.show()

            print(f"Mean angle error: {np.mean(angle_error)}")
            print(f"Mean norm of y-y_temp: {np.mean(error_y)}")

        # Saving the location of the marker
        loc = (marker_o + marker_d + marker_x + marker_y) / 4
        R_list = np.stack([x_axis, y_axis, z_axis], axis=-1)
        return WorldTrace(timestamps, loc, R_list)

    @staticmethod
    def construct_from_markers_robust(timestamps: np.ndarray, marker_o: np.ndarray, marker_d: np.ndarray, marker_x: np.ndarray,
                                      marker_y: np.ndarray, threshold: float = 5.0) -> 'WorldTrace':
        """
        Robust version of construct_from_markers that detects when one of the 4 markers
        is faulty (e.g. flipped, occluded) using the Marker Fault Isolation Metric, and
        reconstructs the rigid body coordinate system using only the remaining 3 healthy markers.
        Fully vectorized via NumPy for high-performance execution.
        """
        N = len(timestamps)
        
        pos = [marker_o, marker_d, marker_x, marker_y]
        d = {}
        bar_d = {}
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in pairs:
            d[(i, j)] = np.linalg.norm(pos[i] - pos[j], axis=1)
            bar_d[(i, j)] = np.median(d[(i, j)])
            
        e = {}
        for i, j in pairs:
            e[(i, j)] = np.abs(d[(i, j)] - bar_d[(i, j)])
            
        epsilon = 0.001 # 1mm regularization
        fault_scores = np.zeros((4, N))
        for i in range(4):
            num_pairs = [p for p in pairs if i in p]
            numerator = np.sum([e[p] for p in num_pairs], axis=0)
            den_pairs = [p for p in pairs if i not in p]
            denominator = np.sum([e[p] for p in den_pairs], axis=0)
            fault_scores[i] = numerator / (denominator + epsilon)
            
        # Nominal plate dimensions (medians over the whole trial)
        w = (bar_d[(1, 2)] + bar_d[(0, 3)]) / 2.0  # (||X-D|| + ||O-Y||) / 2
        h = (bar_d[(0, 2)] + bar_d[(1, 3)]) / 2.0  # (||O-X|| + ||Y-D||) / 2

        # Precompute Case 0 (O is faulty)
        x_0 = marker_x - marker_d
        x_0 = x_0 / np.linalg.norm(x_0, axis=1)[:, None]
        yt_0 = marker_y - marker_d
        yt_0 = yt_0 / np.linalg.norm(yt_0, axis=1)[:, None]
        z_0 = np.cross(x_0, yt_0)
        z_0 = z_0 / np.linalg.norm(z_0, axis=1)[:, None]
        y_0 = np.cross(z_0, x_0)
        # Estimate O: O_est = D + w*x + h*y
        o_est = marker_d + w * x_0 + h * y_0
        loc_0 = (o_est + marker_d + marker_x + marker_y) / 4.0
        rot_0 = np.stack((x_0, y_0, z_0), axis=2)

        # Precompute Case 1 (D is faulty)
        x_1 = marker_o - marker_y
        x_1 = x_1 / np.linalg.norm(x_1, axis=1)[:, None]
        yt_1 = marker_o - marker_x
        yt_1 = yt_1 / np.linalg.norm(yt_1, axis=1)[:, None]
        z_1 = np.cross(x_1, yt_1)
        z_1 = z_1 / np.linalg.norm(z_1, axis=1)[:, None]
        y_1 = np.cross(z_1, x_1)
        # Estimate D: D_est = O - w*x - h*y
        d_est = marker_o - w * x_1 - h * y_1
        loc_1 = (marker_o + d_est + marker_x + marker_y) / 4.0
        rot_1 = np.stack((x_1, y_1, z_1), axis=2)

        # Precompute Case 2 (X is faulty)
        x_2 = marker_o - marker_y
        x_2 = x_2 / np.linalg.norm(x_2, axis=1)[:, None]
        yt_2 = marker_y - marker_d
        yt_2 = yt_2 / np.linalg.norm(yt_2, axis=1)[:, None]
        z_2 = np.cross(x_2, yt_2)
        z_2 = z_2 / np.linalg.norm(z_2, axis=1)[:, None]
        y_2 = np.cross(z_2, x_2)
        # Estimate X: X_est = Y + w*x - h*y
        x_est = marker_y + w * x_2 - h * y_2
        loc_2 = (marker_o + marker_d + x_est + marker_y) / 4.0
        rot_2 = np.stack((x_2, y_2, z_2), axis=2)

        # Precompute Case 3 (Y is faulty)
        x_3 = marker_x - marker_d
        x_3 = x_3 / np.linalg.norm(x_3, axis=1)[:, None]
        yt_3 = marker_o - marker_x
        yt_3 = yt_3 / np.linalg.norm(yt_3, axis=1)[:, None]
        z_3 = np.cross(x_3, yt_3)
        z_3 = z_3 / np.linalg.norm(z_3, axis=1)[:, None]
        y_3 = np.cross(z_3, x_3)
        # Estimate Y: Y_est = X - w*x + h*y
        y_est = marker_x - w * x_3 + h * y_3
        loc_3 = (marker_o + marker_d + marker_x + y_est) / 4.0
        rot_3 = np.stack((x_3, y_3, z_3), axis=2)

        # Precompute Case 4 (Normal - all 4 markers)
        x_4_1 = marker_x - marker_d
        x_4_1 = x_4_1 / np.linalg.norm(x_4_1, axis=1)[:, None]
        x_4_2 = marker_o - marker_y
        x_4_2 = x_4_2 / np.linalg.norm(x_4_2, axis=1)[:, None]
        x_4 = (x_4_1 + x_4_2) / 2.0
        x_4 = x_4 / np.linalg.norm(x_4, axis=1)[:, None]
        
        y_4_1 = marker_o - marker_x
        y_4_1 = y_4_1 / np.linalg.norm(y_4_1, axis=1)[:, None]
        y_4_2 = marker_y - marker_d
        y_4_2 = y_4_2 / np.linalg.norm(y_4_2, axis=1)[:, None]
        yt_4 = (y_4_1 + y_4_2) / 2.0
        yt_4 = yt_4 / np.linalg.norm(yt_4, axis=1)[:, None]
        
        z_4 = np.cross(x_4, yt_4)
        z_4 = z_4 / np.linalg.norm(z_4, axis=1)[:, None]
        y_4 = np.cross(z_4, x_4)
        loc_4 = (marker_o + marker_d + marker_x + marker_y) / 4.0
        rot_4 = np.stack((x_4, y_4, z_4), axis=2)

        # Determine Case for each timestep
        max_idx = np.argmax(fault_scores, axis=0)
        is_faulty = np.max(fault_scores, axis=0) > threshold
        case_indices = np.where(is_faulty, max_idx, 4)

        # advanced index to select correct cases
        all_rotations = np.stack((rot_0, rot_1, rot_2, rot_3, rot_4), axis=0)
        all_positions = np.stack((loc_0, loc_1, loc_2, loc_3, loc_4), axis=0)
        
        rotations = all_rotations[case_indices, np.arange(N)]
        positions = all_positions[case_indices, np.arange(N)]
        
        # Alignment warnings (same check as standard construction)
        y_axis = rotations[:, :, 1]
        y_axis_temp_1 = marker_o - marker_x
        y_axis_temp_1 = y_axis_temp_1 / np.linalg.norm(y_axis_temp_1, axis=1)[:, None]
        
        error_y = np.linalg.norm(y_axis - y_axis_temp_1, axis=1)
        angle_error = np.arccos(np.clip(np.sum(y_axis * y_axis_temp_1, axis=1), -1, 1)) * 180 / np.pi
        
        if np.mean(angle_error) > 1.0:
            print(f"Mean angle error: {np.mean(angle_error)}")
            print(f"Mean norm of y-y_temp: {np.mean(error_y)}")
            
        return WorldTrace(timestamps, positions, rotations)

        
    @staticmethod
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
        pos_x = _generate_smooth_motion_profile(num_samples, duration, max_amp=0.5)
        pos_y = _generate_smooth_motion_profile(num_samples, duration, max_amp=0.3)
        pos_z = _generate_smooth_motion_profile(num_samples, duration, max_amp=0.5)
        positions = np.column_stack((pos_x, pos_y, pos_z))

        # --- Generate smooth random orientation ---
        # Create motion profiles for Euler angles
        rot_z = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi)
        rot_y = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi / 2)
        rot_x = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi / 2)

        # Convert Euler angles to a stack of rotation matrices
        rotations_obj = Rotation.from_euler('zyx', np.vstack([rot_z, rot_y, rot_x]).T)
        rotations = rotations_obj.as_matrix()

        return WorldTrace(timestamps, positions, rotations)
    
    def get_sample_frequency(self):
        """
        This function returns the sample frequency of the WorldTrace.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def lowpass_filter(self, cutoff_freq: float, order: int):
        """
        This function applies a lowpass filter to the WorldTrace.
        """
        sample_freq = self.get_sample_frequency()
        nyquist_freq = 0.5 * sample_freq
        cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, cutoff, btype='low') # type: ignore
        positions = filtfilt(b, a, self.positions, axis=0)
        angle_axis = Rotation.from_matrix(self.rotations).as_rotvec()
        angle_axis = filtfilt(b, a, angle_axis, axis=0)
        rotations = Rotation.from_rotvec(angle_axis).as_matrix()
        return WorldTrace(self.timestamps, positions, rotations)

    def get_rotation_errors_deg(self, other_trace: 'WorldTrace') -> np.ndarray:
        """
        This function returns a time series list of the rotation errors in degrees between two WorldTrace instances.
        """
        assert len(self) == len(other_trace), "WorldTraces must have the same length to compare them."
        
        errors = Rotation.from_matrix(np.matmul(self.rotations.transpose(0, 2, 1), other_trace.rotations))
        angle_axis = errors.as_rotvec()
        angle_deg = np.linalg.norm(angle_axis, axis=1) * 180.0 / np.pi
        return angle_deg

    def get_joint_center(self, other_world_trace: 'WorldTrace') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Given two world traces, solve for the best fit constant offset from a joint center.
        This is done by minimizing the sum of the squared differences between the two traces. """
        # Parent is other, child is self
        assert isinstance(other_world_trace, WorldTrace), "Must pass a WorldTrace instance to compare."
        assert len(self) == len(other_world_trace), "WorldTraces must have the same length to compare them."

        parent_loc = self.positions
        child_loc = other_world_trace.positions

        r_c_p = parent_loc - child_loc
        r_c_p = r_c_p.flatten()

        R_w_parent = self.rotations.reshape(-1, 3)
        R_w_child = other_world_trace.rotations.reshape(-1, 3)
        R_w = np.hstack((-R_w_parent, R_w_child))

        offsets, res, rank, S = np.linalg.lstsq(R_w, r_c_p, rcond=None)

        parent_offset = offsets[:3]
        child_offset = offsets[3:]
        error = R_w_parent @ parent_offset - R_w_child @ child_offset + r_c_p
        error = error.reshape(-1, 3)
        return parent_offset, child_offset, error
    
    def get_primary_joint_axis(self, other_world_trace: 'WorldTrace') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the primary axis of a hinge joint between two world traces using
        the Mean Axis method, expressed in both the parent's ('self') and
        child's ('other') coordinate frames.

        This method works by finding the axis of rotation for the relative
        orientation at each time step and then determining the mean direction of
        that axis over the entire trial. This is robust to joints where the
        instantaneous axis of rotation may wobble. The mean direction is found
        by computing the principal eigenvector of the covariance matrix of the
        instantaneous axes.

        Args:
            other_world_trace (WorldTrace): The 'child' segment's world trace.
                                          'self' is assumed to be the 'parent'.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - axis_in_self (np.ndarray): The (3,) unit vector for the primary
                                         axis in the `self` (parent) frame.
            - axis_in_other (np.ndarray): The (3,) unit vector for the primary
                                          axis in the `other` (child) frame.
        """
        assert isinstance(other_world_trace, WorldTrace), "Must pass a WorldTrace instance."
        assert len(self) == len(other_world_trace), "WorldTraces must have the same length."

        def _find_mean_axis(relative_rotations: Rotation) -> np.ndarray:
            """
            Finds the mean axis of rotation from a time series of rotations using
            Principal Component Analysis on the instantaneous axes.
            """
            # Get the rotation vectors (axis * angle) for each time step
            rot_vecs = relative_rotations.as_rotvec()  # shape (N, 3)
            
            # Normalize each rotation vector to get the instantaneous axis of rotation
            # Handle cases where the angle is zero to avoid division by zero
            norms = np.linalg.norm(rot_vecs, axis=1)
            non_zero_mask = norms > 1e-8
            
            # If there's no significant rotation anywhere, we can't determine an axis.
            if not np.any(non_zero_mask):
                # Return a default axis, as no motion was detected.
                return np.array([0., 0., 1.])

            axes = np.zeros_like(rot_vecs)
            # Normalize only the non-zero rotation vectors
            axes[non_zero_mask] = rot_vecs[non_zero_mask] / norms[non_zero_mask, np.newaxis]
            
            # To handle the axis ambiguity (v is the same as -v), we ensure all
            # axes point in the same general direction as the first axis.
            first_axis = axes[np.argmax(non_zero_mask)]
            for i in range(len(axes)):
                if np.dot(axes[i], first_axis) < 0:
                    axes[i] *= -1
            
            # The mean axis is the principal component of the distribution of axes,
            # which is the eigenvector of the covariance matrix corresponding to the
            # largest eigenvalue.
            covariance_matrix = np.cov(axes[non_zero_mask].T)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            
            # The mean axis is the eigenvector with the largest eigenvalue
            mean_axis = eigenvectors[:, np.argmax(eigenvalues)].real
            return mean_axis / np.linalg.norm(mean_axis)

        # --- Step 1: Get Relative Rotations from both perspectives ---
        R_wp_stack = Rotation.from_matrix(self.rotations)
        R_wc_stack = Rotation.from_matrix(other_world_trace.rotations)

        # --- Step 2: Calculate axis in the 'self' (parent) frame ---
        # Use R_pc = R_parent.inv() * R_child
        R_pc_stack = R_wp_stack.inv() * R_wc_stack
        axis_in_self = _find_mean_axis(R_pc_stack)

        # --- Step 3: Calculate axis in the 'other' (child) frame ---
        # Use R_cp = R_child.inv() * R_parent
        R_cp_stack = R_wc_stack.inv() * R_wp_stack
        axis_in_other = _find_mean_axis(R_cp_stack)

        return axis_in_self, axis_in_other
    
