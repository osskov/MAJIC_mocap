import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Union
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.interpolate import interp1d

# Assuming these utilities are in a relative path
from .finite_difference_utils import central_difference, forward_difference, polynomial_fit_derivative
from .gyro_utils import calculate_best_fit_rotation


class IMUTrace:
    """
    A container for a time-series trace of IMU data.

    This class stores synchronized gyroscope, accelerometer, and magnetometer
    data along with their corresponding timestamps. It provides utility methods
    for common operations like slicing, resampling, coordinate projection,
    and comparison.

    Attributes:
        timestamps (np.ndarray): A 1D array of N timestamps, typically in seconds.
        gyro (np.ndarray): An (N, 3) numpy array representing
            angular velocity (e.g., in rad/s) at each timestamp.
        acc (np.ndarray): An (N, 3) numpy array representing
            linear acceleration (e.g., in m/s^2) at each timestamp.
        mag (np.ndarray): An (N, 3) numpy array representing
            magnetic field data (e.g., in arbitrary units or Gauss) at each timestamp.
    """
    timestamps: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    mag: np.ndarray

    def __init__(self, timestamps: np.ndarray, gyro: Union[List[np.ndarray], np.ndarray], acc: Union[List[np.ndarray], np.ndarray], mag: Union[List[np.ndarray], np.ndarray]):
        """
        Initializes the IMUTrace object.

        Args:
            timestamps (np.ndarray): 1D array of timestamps.
            gyro (Union[List[np.ndarray], np.ndarray]): Gyro data array of shape (N, 3).
            acc (Union[List[np.ndarray], np.ndarray]): Accelerometer data array of shape (N, 3).
            mag (Union[List[np.ndarray], np.ndarray]): Magnetometer data array of shape (N, 3).

        Raises:
            AssertionError: If the lengths of timestamps, gyro, acc, and mag
            do not all match.
        """
        assert (len(timestamps) == len(gyro) == len(acc) == len(mag)), \
            "All data streams (timestamps, gyro, acc, mag) must have the same length."
        self.timestamps = timestamps
        self.gyro = np.asarray(gyro)
        self.acc = np.asarray(acc)
        self.mag = np.asarray(mag)

    @classmethod
    def from_txt(cls, file_path: Union[str, Path]) -> 'IMUTrace':
        """Parses a single Xsens-formatted IMU .txt file."""
        file_path = Path(file_path)
        freq = 100.0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for _ in range(10):
                line = f.readline()
                if line.startswith("// Update Rate"):
                    try:
                        freq = float(line.split(":")[1].split("Hz")[0].strip())
                    except (IndexError, ValueError):
                        pass
                    break

        df = pd.read_csv(
            file_path,
            delimiter='\t',
            skiprows=5,
            engine='c',
            usecols=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
        )
        
        timestamps = np.arange(len(df), dtype=np.float64) / freq
        acc = df[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy(dtype=np.float64)
        gyro = df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy(dtype=np.float64)
        mag = df[['Mag_X', 'Mag_Y', 'Mag_Z']].to_numpy(dtype=np.float64)

        return cls(timestamps=timestamps, gyro=gyro, acc=acc, mag=mag)

    @classmethod
    def from_folder(cls, folder_path: Union[str, Path]) -> Dict[str, 'IMUTrace']:
        """Loads all IMU .txt files in a folder directly by their segment names."""
        folder = Path(folder_path)
        imu_dir = folder / 'imu data' if (folder / 'imu data').is_dir() else folder

        imu_files = list(imu_dir.glob("*.txt"))
        imu_traces = {f.name.replace('.txt', ''): cls.from_txt(f) for f in imu_files}

        if not imu_traces:
            raise FileNotFoundError(f"No IMU .txt files found in: {imu_dir}")

        return imu_traces

    def __len__(self):
        """
        Returns the number of samples in the IMUTrace.

        This allows us to call len(trace) on an IMUTrace instance.
        """
        return len(self.timestamps)

    def __getitem__(self, key) -> 'IMUTrace':
        """
        Allows indexing or slicing of the IMUTrace.

        This allows us to slice the IMUTrace instance and access ranges of
        items with `sub_trace = trace[1:4]`. If an integer is passed, it
        returns a new IMUTrace of length 1, e.g., `sub_trace = trace[2]`.

        Args:
            key (slice or int): The index or slice to apply.

        Returns:
            IMUTrace: A new IMUTrace object containing the selected data.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new IMUTrace instance with the sliced items
            return IMUTrace(self.timestamps[key], self.gyro[key], self.acc[key], self.mag[key])
        else:
            # If key is an integer, wrap it in a new IMUTrace of length 1
            # This ensures that the return type is always an IMUTrace
            return IMUTrace(np.array([self.timestamps[key]]), self.gyro[key:key+1], self.acc[key:key+1], self.mag[key:key+1])

    def __eq__(self, other):
        """
        Checks for exact equality between two IMUTrace instances.

        This will return True if the timestamps, gyro, acc, and mag data
        are all *exactly* equal. Uses element-wise comparison.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if all corresponding data points are identical, False otherwise.
        """
        if not isinstance(other, IMUTrace):
            return False
        if len(self) != len(other):
            return False
            
        # Perform element-wise comparison for all data
        return (np.array_equal(self.timestamps, other.timestamps) and
                np.array_equal(self.gyro, other.gyro) and
                np.array_equal(self.acc, other.acc) and
                np.array_equal(self.mag, other.mag))

    def __sub__(self, other):
        """
        Subtracts the data of another IMUTrace from this one.

        This will return a new IMUTrace instance with the gyro, acc, and mag
        data subtracted element-wise. The timestamps must be almost equal.
        The timestamps of the new trace are copied from `self`.

        Args:
            other (IMUTrace): The IMUTrace to subtract.

        Returns:
            IMUTrace: A new IMUTrace containing the difference.

        Raises:
            AssertionError: If `other` is not an IMUTrace, if lengths differ,
                or if timestamps are not sufficiently close.
        """
        assert isinstance(other, IMUTrace), "Subtraction can only be done between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to subtract."
        # Ensure timestamps are synchronized before subtracting sensor data
        np.testing.assert_array_almost_equal(self.timestamps, other.timestamps, decimal=6,
                                             err_msg="IMUTraces must have the same timestamps to subtract.")

        # Subtract sensor data element-wise
        gyro = self.gyro - other.gyro
        acc = self.acc - other.acc
        mag = self.mag - other.mag
        
        # Return new trace with subtracted data and original timestamps
        return IMUTrace(self.timestamps, gyro, acc, mag)

    def allclose(self, other, atol=1e-6):
        """
        Checks for approximate equality between two IMUTrace instances.

        This will return True if the timestamps, gyro, acc, and mag data
        are all approximately equal within the specified absolute tolerance.

        Args:
            other (object): The object to compare with.
            atol (float, optional): The absolute tolerance. Defaults to 1e-6.

        Returns:
            bool: True if all corresponding data points are close, False otherwise.
        """
        if not isinstance(other, IMUTrace):
            return False
        if len(self) != len(other):
            return False
        if self.gyro.shape[0] != other.gyro.shape[0] or self.acc.shape[0] != other.acc.shape[0] or self.mag.shape[0] != other.mag.shape[0]:
            return False
            
        # Check for empty traces (which are considered close)
        if len(self) == 0:
            return True
            
        # Check that the shape of the first element matches
        if self.gyro.shape[1:] != other.gyro.shape[1:] or self.acc.shape[1:] != other.acc.shape[1:] or self.mag.shape[1:] != other.mag.shape[1:]:
            return False
            
        # Use np.allclose for floating-point comparisons
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                np.allclose(self.gyro, other.gyro, atol=atol) and
                np.allclose(self.acc, other.acc, atol=atol) and
                np.allclose(self.mag, other.mag, atol=atol))

    def copy(self) -> 'IMUTrace':
        """
        Returns a deep copy of the IMUTrace object.

        This ensures all underlying numpy arrays (timestamps, gyro, acc, mag)
        and their containing lists are duplicated to prevent unintended
        modification of the original object.

        Returns:
            IMUTrace: A new, independent copy of the IMUTrace.
        """
        return IMUTrace(
            timestamps=self.timestamps.copy(),
            # Create new arrays containing copies of each numpy array
            gyro=self.gyro.copy(),
            acc=self.acc.copy(),
            mag=self.mag.copy()
        )

    def shallow_copy(self) -> 'IMUTrace':
        """
        Returns a shallow copy of the IMUTrace object, sharing the underlying numpy arrays.
        """
        return IMUTrace(
            timestamps=self.timestamps,
            gyro=self.gyro,
            acc=self.acc,
            mag=self.mag
        )

    def _finite_difference_gyros(self, method='polyfit') -> np.ndarray:
        r"""
        Private method to compute the angular acceleration (derivative of gyro).
        
        This can be calculated a number of different ways.

        Args:
            method (str, optional): The finite difference method to use.
                'central': Computes using a central difference.
                'first_order': Computes using a first-order forward difference.
                'polyfit': Computes using a 2nd-order polynomial fit.
                Defaults to 'polyfit'.

        Returns:
            np.ndarray: An (N, 3) numpy array representing
            angular acceleration ($\dot{\omega}$) at each timestamp.
        """
        derivates = []
        
        # Calculate derivatives for each axis (x, y, z) separately
        if method == 'central':
            derivates = [central_difference(self.gyro[:, axis], self.timestamps) for axis in range(3)]
        elif method == 'first_order':
            derivates = [forward_difference(self.gyro[:, axis], self.timestamps) for axis in range(3)]
        elif method == 'polyfit':
            derivates = [polynomial_fit_derivative(self.gyro[:, axis], self.timestamps, order=2) for axis in range(3)]
                
        return np.column_stack(derivates)

    def project_acc(self, local_offset: Union[np.ndarray, List[np.ndarray]],
                    finite_difference_gyro_method='polyfit') -> 'IMUTrace':
        r"""
        Projects acceleration to a new point on the same rigid body.

        This method calculates the linear acceleration at a point defined by
        `local_offset` relative to the IMU's measurement frame. It uses the
        full rigid body acceleration equation:
        $a_p = a_o + \dot{\omega} \times r + \omega \times (\omega \times r)$

        where:
        - $a_p$ is the projected acceleration (the result)
        - $a_o$ is the original measured acceleration (self.acc)
        - $\omega$ is the angular velocity (self.gyro)
        - $\dot{\omega}$ is the angular acceleration (computed from gyro)
        - $r$ is the local_offset vector

        Args:
            local_offset (Union[np.ndarray, List[np.ndarray]]): The 3D position
                vector (or list of vectors) from the IMU measurement frame to the
                new point of interest, expressed in the IMU's local frame.
            finite_difference_gyro_method (str, optional): The method used to
                calculate the angular acceleration ($\dot{\omega}$).
                Passed to `_finite_difference_gyros`. Defaults to 'polyfit'.

        Returns:
            IMUTrace: A new IMUTrace object with the same timestamps, gyro, and
            mag data, but with the `acc` data projected to the new location.
        """
        # If a single offset is given, create a list of that offset
        if isinstance(local_offset, np.ndarray):
            local_offset = [local_offset] * len(self)
            
        # Calculate angular acceleration
        gyro_derivative = self._finite_difference_gyros(finite_difference_gyro_method)
        
        # Apply the rigid body acceleration equation for each time step
        # a_p = a_o + (d_gyro x r) + (gyro x (gyro x r))
        # a_o: Original acceleration (a)
        # d_gyro x r: Tangential acceleration (np.cross(dg, r))
        # gyro x (gyro x r): Centripetal acceleration (np.cross(g, np.cross(g, r)))
        
        tangential_acc = np.cross(gyro_derivative, local_offset)
        centripetal_acc = np.cross(self.gyro, np.cross(self.gyro, local_offset))
        acc_projected = self.acc + tangential_acc + centripetal_acc
            
        return IMUTrace(self.timestamps, self.gyro.copy(), acc_projected, self.mag.copy())

    def add_noise(self,
                  gyro_noise_std: float = 0.0,
                  acc_noise_std: float = 0.0,
                  mag_noise_std: float = 0.0
                  ) -> 'IMUTrace':
        """
        Adds synthetic Gaussian noise to the sensor data.

        This method generates random noise from a normal distribution (mean=0)
        with the specified standard deviation for each sensor type and adds it
        to the trace data. It returns a new IMUTrace object, leaving the
        original unchanged.

        Args:
            gyro_noise_std (float, optional): Standard deviation of the
                gyroscope noise (e.g., in rad/s). Defaults to 0.0.
            acc_noise_std (float, optional): Standard deviation of the
                accelerometer noise (e.g., in m/s^2). Defaults to 0.0.
            mag_noise_std (float, optional): Standard deviation of the
                magnetometer noise (in arbitrary units). Defaults to 0.0.

        Returns:
            IMUTrace: A new IMUTrace object with the added noise.
        """
        # Create a deep copy to avoid modifying the original trace
        noisy_trace = self.copy()
        num_samples = len(self)

        # Add gyroscope noise if a non-zero standard deviation is provided
        if gyro_noise_std > 0:
            gyro_noise = np.random.normal(0, gyro_noise_std, size=noisy_trace.gyro.shape)
            noisy_trace.gyro = noisy_trace.gyro + gyro_noise

        # Add accelerometer noise
        if acc_noise_std > 0:
            acc_noise = np.random.normal(0, acc_noise_std, size=noisy_trace.acc.shape)
            noisy_trace.acc = noisy_trace.acc + acc_noise

        # Add magnetometer noise
        if mag_noise_std > 0:
            mag_noise = np.random.normal(0, mag_noise_std, size=noisy_trace.mag.shape)
            noisy_trace.mag = noisy_trace.mag + mag_noise

        return noisy_trace
    
    def get_sample_frequency(self) -> float:
        """
        Calculates and returns the average sample frequency of the trace.

        The frequency is computed as the inverse of the mean difference
        between consecutive timestamps.

        Returns:
            float: The average sample frequency in Hz.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def calculate_gyro_angle_error(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculates the pointwise angle error between two gyroscope traces.

        This method computes the angle (in radians) between the angular velocity
        vectors from `self` and `other` at each corresponding timestamp.

        Args:
            other (IMUTrace): Another IMUTrace object to compare against.

        Returns:
            np.ndarray: A 1D array containing the angle error (in radians)
            at each sample. Returns 0 if both vectors are near-zero,
            and np.nan if only one is near-zero.

        Raises:
            AssertionError: If `other` is not an IMUTrace or if the
            traces have different lengths.
        """
        assert isinstance(other, IMUTrace), "Angle error can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate angle error."

        norm_self = np.linalg.norm(self.gyro, axis=1)
        norm_other = np.linalg.norm(other.gyro, axis=1)
        
        # Calculate dot product where possible
        dot_product = np.sum(self.gyro * other.gyro, axis=1)
        
        angle_error = np.zeros(len(self))
        
        # Avoid division by zero
        valid_mask = (norm_self > 1e-8) & (norm_other > 1e-8)
        zero_mask = (norm_self < 1e-8) & (norm_other < 1e-8)
        
        # Case 1: Both vectors have significant magnitude
        cos_theta = dot_product[valid_mask] / (norm_self[valid_mask] * norm_other[valid_mask])
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_error[valid_mask] = np.arccos(cos_theta)
        
        # Case 2: Both vectors are near-zero
        angle_error[zero_mask] = 0.0
        
        # Case 3: Only one vector is zero (angle is undefined)
        invalid_mask = ~(valid_mask | zero_mask)
        angle_error[invalid_mask] = np.nan
                
        return angle_error

    def calculate_rotation_offset_from_gyros(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculates the best-fit static rotation offset between two gyro traces.

        This method finds a single 3x3 rotation matrix (R_self_other) that
        best aligns the gyroscope data from `other` to `self` over the entire
        trace. This is typically used to find the relative orientation
        between two IMUs that are rigidly attached.

        Assumes: `self.gyro[i] \approx R_self_other @ other.gyro[i]`

        Args:
            other (IMUTrace): The other IMUTrace object (representing the
                'other' coordinate frame).

        Returns:
            np.ndarray: A 3x3 rotation matrix (R_self_other) that rotates
            vectors from the 'other' frame to the 'self' frame.

        Raises:
            AssertionError: If `other` is not an IMUTrace or if the
            traces have different lengths.
        """
        assert isinstance(other, IMUTrace), "Rotation offset can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate rotation offset."

        # Delegate to the external utility function
        R_so = calculate_best_fit_rotation(self.gyro, other.gyro)
        return R_so

    def re_zero_timestamps(self) -> 'IMUTrace':
        """
        Creates a new IMUTrace with timestamps starting from zero.

        This shifts the entire timestamp vector by subtracting the
        first timestamp value (`self.timestamps[0]`).

        Returns:
            IMUTrace: A new IMUTrace object with re-zeroed timestamps.
        """
        return IMUTrace(self.timestamps - self.timestamps[0], self.gyro, self.acc, self.mag)

    def resample(self, new_frequency: float):
        """
        Resamples the IMU trace to a new, constant frequency.

        This method uses linear interpolation (`scipy.interpolate.interp1d`)
        to resample the gyro, accelerometer, and magnetometer data to a
        new set of timestamps defined by the `new_frequency`.

        The new time vector starts at the original `self.timestamps[0]` and
        ends at or slightly after `self.timestamps[-1]`.

        Args:
            new_frequency (float): The target sample frequency in Hz.

        Returns:
            IMUTrace: A new, resampled IMUTrace object.
        """
        new_dt = 1 / new_frequency
        old_dt = np.mean(np.diff(self.timestamps))
        
        # If frequency is already correct, just return a copy
        if np.isclose(new_dt, old_dt):
            return self.copy()

        # Create new timestamp vector
        # Add a small buffer to ensure the last sample is included
        stop_time = self.timestamps[-1] + min(old_dt, new_dt)
        new_timestamps = np.arange(start=self.timestamps[0], stop=stop_time, step=new_dt)
        
        # Create linear interpolators for each data type
        # 'extrapolate' is used to handle requests slightly outside the original time range
        gyro_interpolator = interp1d(self.timestamps, self.gyro, axis=0, kind='linear',
                                     fill_value='extrapolate')
        acc_interpolator = interp1d(self.timestamps, self.acc, axis=0, kind='linear',
                                    fill_value='extrapolate')
        mag_interpolator = interp1d(self.timestamps, self.mag, axis=0, kind='linear',
                                    fill_value='extrapolate')

        # Get new resampled data as monolithic (N_new, 3) arrays
        new_gyro = gyro_interpolator(new_timestamps)
        new_acc = acc_interpolator(new_timestamps)
        new_mag = mag_interpolator(new_timestamps)

        return IMUTrace(new_timestamps, new_gyro, new_acc, new_mag)

    
    def find_spheroidal_joint_offset(self,
                                     other: 'IMUTrace',
                                     initial_offset_self: np.ndarray = None,
                                     initial_offset_other: np.ndarray = None,
                                     max_iterations: int = 20,
                                     tolerance: float = 1e-6,
                                     subsample_rate: int = 1) -> Dict[str, Union[np.ndarray, bool]]:
        r"""
        Estimates the position of a spheroidal joint center relative to two IMUs
        using a vectorized Gauss-Newton optimization.

        This algorithm minimizes the difference in the *magnitude* of the
        projected joint center acceleration, as described in:
        
        Seel, T. (2014) Joint axis and position estimation from inertial measurement
        data by exploiting kinematic constraints.

        The cost function is $e = ||a_{p1}|| - ||a_{p2}||$, where
        $a_p = a_o - (\dot{\omega} \times o + \omega \times (\omega \times o))$.

        Args:
            other (IMUTrace): The IMUTrace object for the other body segment.
            initial_offset_self (np.ndarray, optional): A (3,) array for the
                initial guess of the offset (o1). Defaults to a small random vector.
            initial_offset_other (np.ndarray, optional): A (3,) array for the
                initial guess of the offset (o2). Defaults to a small random vector.
            max_iterations (int, optional): Max iterations for the optimization.
            tolerance (float, optional): Convergence tolerance based on the
                norm of the update step (delta_x).
            subsample_rate (int, optional): Rate to subsample data for performance.
                `1` uses all data, `10` uses every 10th sample.

        Returns:
            Dict[str, Union[np.ndarray, bool]]: A dictionary containing:
                - 'offset_self': The optimized (3,) offset vector (o1) in self.frame.
                - 'offset_other': The optimized (3,) offset vector (o2) in other.frame.
                - 'converged': A boolean flag indicating if convergence was reached.
        """
        # Ensure data streams are synchronized (timestamps must match)
        np.testing.assert_array_almost_equal(
            self.timestamps, other.timestamps, decimal=5,
            err_msg="IMU traces must be time-synchronized. Use the resample() method first."
        )

        # Pre-compute gyro derivatives (angular acceleration) for all samples
        # This returns (N_full, 3) NumPy arrays
        g1_dot_full = self._finite_difference_gyros()
        g2_dot_full = other._finite_difference_gyros()

        # Subsample data to speed up computation.
        indices = np.arange(0, len(self), subsample_rate)
        g1 = self.gyro[indices]    # (N, 3)
        g2 = other.gyro[indices]   # (N, 3)
        a1 = self.acc[indices]     # (N, 3)
        a2 = other.acc[indices]    # (N, 3)
        g1_dot = g1_dot_full[indices]        # (N, 3)
        g2_dot = g2_dot_full[indices]        # (N, 3)

        # Initialize the state vector x = [o1_x, o1_y, o1_z, o2_x, o2_y, o2_z]
        o1 = np.random.rand(3) * 0.1 if initial_offset_self is None else initial_offset_self.copy()
        o2 = np.random.rand(3) * 0.1 if initial_offset_other is None else initial_offset_other.copy()
        x = np.concatenate([o1, o2])

        # Gauss-Newton optimization loop
        for _ in range(max_iterations):
            # Unpack current offset estimates
            o1, o2 = x[:3], x[3:]

            # --- Vectorized Computation (Replaces the inner for-loop) ---
            # All operations are now on (N, 3) or (N,) arrays, where N=len(indices)

            # Eq(3): Γ_g(o) = g x (g x o) + ġ x o
            # This is the contribution of angular motion to linear acceleration
            # Broadcasting: (N, 3) x (3,) -> (N, 3)
            gamma1 = np.cross(g1, np.cross(g1, o1)) + np.cross(g1_dot, o1) # (N, 3)
            gamma2 = np.cross(g2, np.cross(g2, o2)) + np.cross(g2_dot, o2) # (N, 3)

            # Calculate projected joint center acceleration: a_p = a_o - Γ
            joint_acc1 = a1 - gamma1 # (N, 3)
            joint_acc2 = a2 - gamma2 # (N, 3)

            # Calculate norm of projected acceleration: ||a_p||
            norm1 = np.linalg.norm(joint_acc1, axis=1) # (N,)
            norm2 = np.linalg.norm(joint_acc2, axis=1) # (N,)

            # Residual vector e = ||a_p1|| - ||a_p2||
            e = norm1 - norm2 # (N,)

            # --- Calculate Jacobian (J) ---
            
            # Eq(4) Jacobian helper: Γ_T(v) = (v x g) x g + v x ġ
            v1 = joint_acc1
            gamma_T1_v1 = np.cross(np.cross(v1, g1), g1) + np.cross(v1, g1_dot) # (N, 3)

            v2 = joint_acc2
            gamma_T2_v2 = np.cross(np.cross(v2, g2), g2) + np.cross(v2, g2_dot) # (N, 3)

            # Create "safe" denominators to avoid division by zero
            norm1_safe = np.maximum(norm1, 1e-9)
            norm2_safe = np.maximum(norm2, 1e-9)

            # Jacobian component for o1: d(e)/d(o1) = d(||a_p1||)/d(o1)
            # d(||v||)/d(o) = -Γ_T(v) / ||v||
            J_o1 = -gamma_T1_v1 / norm1_safe[:, np.newaxis] # (N, 3)

            # Jacobian component for o2: d(e)/d(o2) = -d(||a_p2||)/d(o2)
            # d(-||v||)/d(o) = +Γ_T(v) / ||v||
            J_o2 =  gamma_T2_v2 / norm2_safe[:, np.newaxis] # (N, 3)

            # Assemble full (N, 6) Jacobian
            J = np.hstack([J_o1, J_o2])

            # Vectorized safety check: Find all rows where either norm was too small
            invalid_mask = (norm1 < 1e-9) | (norm2 < 1e-9)
            # Zero out the residual and Jacobian for these rows
            e[invalid_mask] = 0.0
            J[invalid_mask, :] = 0.0

            # --- End Vectorized Computation ---

            try:
                # Calculate update step: delta_x = -J_pseudo_inv * e
                # (6, N) @ (N,) -> (6,)
                delta_x = -np.linalg.pinv(J) @ e
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in pseudoinverse calculation. Stopping iteration.")
                return {'offset_self': x[:3], 'offset_other': x[3:], 'converged': False}

            # Update state vector
            x += delta_x

            # Check for convergence
            if np.linalg.norm(delta_x) < tolerance:
                return {'offset_self': x[:3], 'offset_other': x[3:], 'converged': True}

        # If loop finishes without converging
        return {'offset_self': x[:3], 'offset_other': x[3:], 'converged': False}


    def find_hinge_joint_axis(self,
                              other: 'IMUTrace',
                              initial_axis_self: np.ndarray = None,
                              initial_axis_other: np.ndarray = None,
                              max_iterations: int = 20,
                              tolerance: float = 1e-6,
                              subsample_rate: int = 1) -> Dict[str, Union[np.ndarray, bool]]:
        """
        Estimates the axis of a hinge joint relative to two IMUs using a
        vectorized Gauss-Newton optimization.

        This algorithm minimizes the difference in the *magnitude* of the
        gyroscope component perpendicular to the joint axis, as described in:

        Seel, T. (2014) Joint axis and position estimation from inertial measurement
        data by exploiting kinematic constraints.
        
        The cost function is $e = ||g_1 \times j_1|| - ||g_2 \times j_2||$, where
        $j$ is the (unit) joint axis vector.

        Args:
            other (IMUTrace): The IMUTrace object for the other body segment.
            initial_axis_self (np.ndarray, optional): A (3,) array for the
                initial guess of the axis (j1). Will be normalized.
                Defaults to a random vector.
            initial_axis_other (np.ndarray, optional): A (3,) array for the
                initial guess of the axis (j2). Will be normalized.
                Defaults to a random vector.
            max_iterations (int, optional): Max iterations for the optimization.
            tolerance (float, optional): Convergence tolerance based on the
                norm of the update step (delta_x).
            subsample_rate (int, optional): Rate to subsample data for performance.

        Returns:
            Dict[str, Union[np.ndarray, bool]]: A dictionary containing:
                - 'axis_self': The optimized (3,) unit axis vector (j1) in self.frame.
                - 'axis_other': The optimized (3,) unit axis vector (j2) in other.frame.
                - 'converged': A boolean flag indicating if convergence was reached.
        """
        # Ensure data streams are synchronized (timestamps must match)
        np.testing.assert_array_almost_equal(
            self.timestamps, other.timestamps, decimal=5,
            err_msg="IMU traces must be time-synchronized. Use the resample() method first."
        )

        # Subsample data: Create (N, 3) NumPy arrays
        indices = np.arange(0, len(self), subsample_rate)
        g1 = np.array(self.gyro)[indices]
        g2 = np.array(other.gyro)[indices]

        # --- Helper functions for Eq(5) spherical coordinate parametrization ---
        # (These are only called once per iteration)
        def spherical_to_cartesian(phi, theta):
            """Converts spherical (phi, theta) to Cartesian (x, y, z) unit vector."""
            return np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])

        def cartesian_derivatives(phi, theta):
            """Calculates partial derivatives of spherical_to_cartesian."""
            dj_dphi = np.array([-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)])
            dj_dtheta = np.array([-np.cos(phi) * np.sin(theta), np.cos(phi) * np.cos(theta), 0])
            return dj_dphi, dj_dtheta
        # --- End Helper functions ---

        # Initialize state vector x = [phi1, theta1, phi2, theta2]
        if initial_axis_self is None:
            # Random initial guess
            phi1, theta1 = np.random.uniform(-np.pi / 2, np.pi / 2), np.random.uniform(-np.pi, np.pi)
        else:
            # Convert Cartesian guess to spherical coordinates
            j1_init = initial_axis_self / np.linalg.norm(initial_axis_self)
            phi1 = np.arcsin(j1_init[2])
            theta1 = np.arctan2(j1_init[1], j1_init[0])

        if initial_axis_other is None:
            # Random initial guess
            phi2, theta2 = np.random.uniform(-np.pi / 2, np.pi / 2), np.random.uniform(-np.pi, np.pi)
        else:
            # Convert Cartesian guess to spherical coordinates
            j2_init = initial_axis_other / np.linalg.norm(initial_axis_other)
            phi2 = np.arcsin(j2_init[2])
            theta2 = np.arctan2(j2_init[1], j2_init[0])

        x = np.array([phi1, theta1, phi2, theta2])

        # Gauss-Newton optimization loop
        for _ in range(max_iterations):
            phi1, theta1, phi2, theta2 = x
            
            # Get current Cartesian axis estimates
            j1 = spherical_to_cartesian(phi1, theta1)
            j2 = spherical_to_cartesian(phi2, theta2)
            
            # Get partial derivatives for the chain rule
            dj1_dphi1, dj1_dtheta1 = cartesian_derivatives(phi1, theta1)
            dj2_dphi2, dj2_dtheta2 = cartesian_derivatives(phi2, theta2)
            
            # --- Vectorized Computation (Replaces the inner for-loop) ---

            # v = g x j (This is the component of gyro perpendicular to the axis)
            # (N, 3) x (3,) -> (N, 3)
            v1 = np.cross(g1, j1)
            v2 = np.cross(g2, j2)

            # norm = ||v||
            # norm((N, 3), axis=1) -> (N,)
            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)

            # Eq(1) Error: e = ||g1 x j1|| - ||g2 x j2||
            # (N,) - (N,) -> (N,)
            e = norm1 - norm2
            
            # Handle potential division by zero
            norm1_safe = np.maximum(norm1, 1e-9)
            norm2_safe = np.maximum(norm2, 1e-9)
            
            # --- Calculate Jacobian (J) ---
            
            # Eq(2) Gradient: d(||g x j||)/dj = ((g x j) x g) / ||g x j||
            # (N, 3) x (N, 3) -> (N, 3)
            # (N, 3) / (N, 1) -> (N, 3)
            d_norm1_dj1 = np.cross(v1, g1) / norm1_safe[:, np.newaxis]
            d_norm2_dj2 = np.cross(v2, g2) / norm2_safe[:, np.newaxis]

            # Chain rule for Jacobian columns
            # d(e)/d(phi1) = (d(e)/d(j1)) @ (d(j1)/d(phi1))
            # (N, 3) @ (3,) -> (N,)
            J_col0 = d_norm1_dj1 @ dj1_dphi1
            J_col1 = d_norm1_dj1 @ dj1_dtheta1
            
            # Note the negative sign: d(e)/d(phi2) = - (d(norm2)/d(j2)) @ (d(j2)/d(phi2))
            J_col2 = - (d_norm2_dj2 @ dj2_dphi2)
            J_col3 = - (d_norm2_dj2 @ dj2_dtheta2)

            # Assemble full (N, 4) Jacobian
            J = np.column_stack((J_col0, J_col1, J_col2, J_col3))

            # Vectorized safety check: Find rows where either norm was too small
            invalid_mask = (norm1 < 1e-9) | (norm2 < 1e-9)
            # Zero out the residual and Jacobian for these rows
            e[invalid_mask] = 0.0
            J[invalid_mask, :] = 0.0
            
            # --- End Vectorized Computation ---

            try:
                # Calculate update step: delta_x = -J_pseudo_inv * e
                # (4, N) @ (N,) -> (4,)
                delta_x = -np.linalg.pinv(J) @ e
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in pseudoinverse calculation. Stopping iteration.")
                j1, j2 = spherical_to_cartesian(x[0], x[1]), spherical_to_cartesian(x[2], x[3])
                return {'axis_self': j1, 'axis_other': j2, 'converged': False}

            # Update state vector (in spherical coordinates)
            x += delta_x

            # Check for convergence
            if np.linalg.norm(delta_x) < tolerance:
                j1, j2 = spherical_to_cartesian(x[0], x[1]), spherical_to_cartesian(x[2], x[3])
                return {'axis_self': j1, 'axis_other': j2, 'converged': True}

        # If loop finishes, return final (non-converged) estimate
        j1, j2 = spherical_to_cartesian(x[0], x[1]), spherical_to_cartesian(x[2], x[3])
        return {'axis_self': j1, 'axis_other': j2, 'converged': False}