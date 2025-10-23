import os
import numpy as np
from typing import List, Dict, Union
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.interpolate import interp1d

from .finite_difference_utils import central_difference, forward_difference, polynomial_fit_derivative
from .gyro_utils import calculate_best_fit_rotation

class IMUTrace:
    timestamps: np.ndarray
    gyro: List[np.ndarray]
    acc: List[np.ndarray]
    mag: List[np.ndarray]

    def __init__(self, timestamps: np.ndarray, gyro: List[np.ndarray], acc: List[np.ndarray], mag: List[np.ndarray]):
        assert (len(timestamps) == len(gyro) == len(acc))
        self.timestamps = timestamps
        self.gyro = gyro
        self.acc = acc
        self.mag = mag

    def __len__(self):
        """
        Returns the number of samples in the IMUTrace. This allows us to call len(trace) on an IMUTrace instance.
        """
        return len(self.timestamps)

    def __getitem__(self, key) -> 'IMUTrace':
        """
        Allows us to use the square bracket notation to access the IMUTrace instance. This allows us to slice the
        IMUTrace instance and access ranges of items with `sub_trace = trace[1:4]`. If we pass an integer, we can
        return the corresponding item as a length 1 trace with `sub_trace = trace[2]` and `len(sub_trace) == 1`.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new IMUTrace instance with the sliced items
            return IMUTrace(self.timestamps[key], self.gyro[key], self.acc[key], self.mag[key])
        else:
            # If key is an integer, return the corresponding item as a solo list
            return IMUTrace(np.array([self.timestamps[key]]), [self.gyro[key]], [self.acc[key]], [self.mag[key]])

    def __eq__(self, other):
        """
        Allows us to compare two IMUTrace instances for equality. This will return True if the timestamps, gyro, and
        acc are all _exactly_ equal.
        """
        if not isinstance(other, IMUTrace):
            return False
        if len(self) != len(other):
            return False
        return ((self.timestamps == other.timestamps).all() and
                all(np.all(self.gyro[i] == other.gyro[i]) for i in range(len(self.gyro))) and
                all(np.all(self.acc[i] == other.acc[i]) for i in range(len(self.acc))) and
                all(np.all(self.mag[i] == other.mag[i]) for i in range(len(self.mag))))

    def __sub__(self, other):
        """
        Allows us to subtract two IMUTrace instances. This will return a new IMUTrace instance with the gyro and acc
        data subtracted.
        """
        assert isinstance(other, IMUTrace), "Subtraction can only be done between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to subtract."
        np.testing.assert_array_almost_equal(self.timestamps, other.timestamps, decimal=6,
                                             err_msg="IMUTraces must have the same timestamps to subtract.")

        gyro = [gyro1 - gyro2 for gyro1, gyro2 in zip(self.gyro, other.gyro)]
        acc = [acc1 - acc2 for acc1, acc2 in zip(self.acc, other.acc)]
        mag = [mag1 - mag2 for mag1, mag2 in zip(self.mag, other.mag)]
        return IMUTrace(self.timestamps, gyro, acc, mag)

    def allclose(self, other, atol=1e-6):
        """
        Allows us to compare two IMUTrace instances for approximate equality. This will return True if the timestamps,
        gyro, and acc are all approximately equal within the specified tolerance.
        """
        if not isinstance(other, IMUTrace):
            return False
        if len(self) != len(other):
            return False
        if len(self.gyro) != len(other.gyro) or len(self.acc) != len(other.acc) or len(self.mag) != len(other.mag):
            return False
        if self.gyro[0].shape != other.gyro[0].shape or self.acc[0].shape != other.acc[0].shape or self.mag[0].shape != \
                other.mag[0].shape:
            return False
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                all(np.allclose(self.gyro[i], other.gyro[i], atol=atol) for i in range(len(self.gyro))) and
                all(np.allclose(self.acc[i], other.acc[i], atol=atol) for i in range(len(self.acc))) and
                all(np.allclose(self.mag[i], other.mag[i], atol=atol) for i in range(len(self.mag))))
    
    def copy(self) -> 'IMUTrace':
        """
        Returns a deep copy of the IMUTrace object, ensuring all underlying numpy arrays
        (timestamps, gyro, acc, mag) are duplicated to prevent unintended modification of the original.
        """
        return IMUTrace(
            timestamps=self.timestamps.copy(),
            # Deep copy the lists of numpy arrays
            gyro=[g.copy() for g in self.gyro],
            acc=[a.copy() for a in self.acc],
            mag=[m.copy() for m in self.mag]
        )

    def _finite_difference_gyros(self, method='polyfit') -> List[np.ndarray]:
        """
        This is a private method to compute the finite difference of the gyros in the IMUTrace. This can be calculated
        a number of different ways.

        method='central' computes the finite difference using a central difference.
        method='first_order' computes the finite difference using a first order approximation.
        method='polyfit' computes the derivatives using a polynomial fit.
        """
        derivates: List[np.ndarray] = []
        if method == 'central':
            derivates = [central_difference(np.array([gyro[axis] for gyro in self.gyro]), self.timestamps) for axis in
                         range(3)]
        elif method == 'first_order':
            derivates = [forward_difference(np.array([gyro[axis] for gyro in self.gyro]), self.timestamps) for axis in
                         range(3)]
        elif method == 'polyfit':
            derivates = [
                polynomial_fit_derivative(np.array([gyro[axis] for gyro in self.gyro]), self.timestamps, order=2) for
                axis in range(3)]
        # Now we need to flip the axes back to the original shape
        return [np.array([derivate[i] for derivate in derivates]) for i in range(len(self.timestamps))]

    def project_acc(self, local_offset: Union[np.ndarray, List[np.ndarray]],
                    finite_difference_gyro_method='polyfit') -> 'IMUTrace':
        """
        This function projects the gyro and acc data in the IMUTrace by a local offset within the frame.
        """
        if isinstance(local_offset, np.ndarray):
            local_offset = [local_offset] * len(self)
        gyro_derivative = self._finite_difference_gyros(finite_difference_gyro_method)
        acc = [acc + np.cross(d_gyro, offset) + np.cross(gyro, np.cross(gyro, offset)) for acc, gyro, d_gyro, offset in
               zip(self.acc, self.gyro, gyro_derivative, local_offset)]
        return IMUTrace(self.timestamps, self.gyro, acc, self.mag)

    def get_timestamps(self) -> np.ndarray:
        """
        Returns the timestamps.
        """
        return self.timestamps

    def get_sample_frequency(self) -> float:
        """
        Returns the sample frequency of the IMUTrace.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def calculate_gyro_angle_error(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the angle error between two gyroscopes.
        """
        assert isinstance(other, IMUTrace), "RMSD can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate RMSD."

        angle_error = np.zeros(len(self))
        for i in range(len(self)):
            self_gyro = self.gyro[i]
            other_gyro = other.gyro[i]
            if np.linalg.norm(self_gyro) > 0 and np.linalg.norm(other_gyro) > 0:
                dot_product = np.dot(self_gyro, other_gyro)
                dot_product /= np.linalg.norm(self_gyro)
                dot_product /= np.linalg.norm(other_gyro)
                dot_product = np.clip(dot_product, -1, 1)
                angle_error[i] = (np.arccos(dot_product))
            elif np.linalg.norm(self_gyro) < 1e-8 and np.linalg.norm(other_gyro) < 1e-8:
                angle_error[i] = 0
            else:
                angle_error[i] = np.nan
        return angle_error

    def calculate_rotation_offset_from_gyros(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the rotation offset between two IMUTraces. Returns rotation matrix R_self_other.
        """
        assert isinstance(other, IMUTrace), "Rotation offset can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate rotation offset."

        R_so = calculate_best_fit_rotation(self.gyro, other.gyro)
        return R_so

    def re_zero_timestamps(self) -> 'IMUTrace':
        """
        Start timestamps at 0
        """
        return IMUTrace(self.timestamps - self.timestamps[0], self.gyro, self.acc, self.mag)

    def resample(self, new_frequency: float):
        new_dt = 1 / new_frequency
        old_dt = np.mean(np.diff(self.timestamps))
        if np.isclose(new_dt, old_dt):
            # No resampling needed if the frequency is the same
            return IMUTrace(self.timestamps, self.gyro, self.acc, self.mag)

        new_timestamps = np.arange(start=self.timestamps[0], stop=self.timestamps[-1] + min(old_dt, new_dt),
                                   step=new_dt)
        gyro_interpolator = interp1d(self.timestamps, np.vstack(self.gyro), axis=0, kind='linear',
                                     fill_value='extrapolate')
        acc_interpolator = interp1d(self.timestamps, np.vstack(self.acc), axis=0, kind='linear',
                                    fill_value='extrapolate')
        mag_interpolator = interp1d(self.timestamps, np.vstack(self.mag), axis=0, kind='linear',
                                    fill_value='extrapolate')

        new_gyro = gyro_interpolator(new_timestamps)
        new_acc = acc_interpolator(new_timestamps)
        new_mag = mag_interpolator(new_timestamps)

        return IMUTrace(new_timestamps, new_gyro, new_acc, new_mag)

    @staticmethod
    def load_IMUTraces_from_folder(imu_folder_path: str) -> Dict[str, 'IMUTrace']:
        imu_traces = {}

        # Find the mapping xml file
        mapping_file = next((f for f in os.listdir(imu_folder_path) if f.endswith('.xml')), None)
        if mapping_file is None:
            raise FileNotFoundError("No mapping file found in IMU folder")

        # Parse the XML file
        tree = ET.parse(os.path.join(imu_folder_path, mapping_file))
        root = tree.getroot()
        trial_prefix = root.find('.//trial_prefix').text

        # Iterate over each ExperimentalSensor element and load its IMUTrace
        for sensor in root.findall('.//ExperimentalSensor'):
            sensor_name = sensor.get('name').strip()
            name_in_model = sensor.find('name_in_model').text.strip()

            file_name = f"{trial_prefix}{sensor_name}.txt"
            file_path = os.path.join(imu_folder_path, 'xsens', 'LowerExtremity', file_name)
            try:
                # Extract update rate
                with open(file_path, "r") as f:
                    for line in f:
                        if line.startswith("// Update Rate"):
                            freq = float(line.split(":")[1].split("Hz")[0])
                            break

                # Read the file into a DataFrame
                df = pd.read_csv(file_path, delimiter='\t', skiprows=5)
                df = df.apply(pd.to_numeric)

                # Extract data
                timestamps = 1 / freq * np.arange(len(df))
                acc = [np.array(row) for row in df[['Acc_X', 'Acc_Y', 'Acc_Z']].values]
                gyro = [np.array(row) for row in df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values]
                mag = [np.array(row) for row in df[['Mag_X', 'Mag_Y', 'Mag_Z']].values]

                # Create IMUTrace objects
                imu_traces[name_in_model] = IMUTrace(timestamps=timestamps, acc=acc, gyro=gyro, mag=mag)
            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping sensor {sensor_name}.")
        return imu_traces
