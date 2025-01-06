from typing import List, Optional, Dict
import numpy as np

from src.toolchest.finite_difference_utils import finite_difference
from toolchest.PlateTrial import PlateTrial


class ProjectedReading:
    """ Represents a virtual sensor reading, projected from multiple real sensor readings.
    Attributes:
        gyro (np.ndarray): Reading from the gyroscope.
        alpha (np.ndarray): Angular acceleration of the board.
        acc (np.ndarray): Reading from the accelerometer.
        acc_change (np.ndarray): Change in acceleration of the board.
        mag (np.ndarray): Reading from the magnetometer.
        timestamp (float): Timestamp of the reading.
        """

    gyro: np.ndarray
    alpha: np.ndarray
    acc: np.ndarray
    acc_change: np.ndarray
    mag: np.ndarray
    timestamp: float

    def __init__(self, gyro: np.ndarray, acc: np.ndarray, mag: np.ndarray, timestamp: float,
                 alpha: Optional[np.ndarray] = np.zeros(3), acc_change: Optional[np.ndarray] = np.zeros(3)):
        """ Initializes a ProjectedReading object with gyroscope, accelerometer, and magnetometer readings,
        a timestamp, and optionally angular acceleration and acceleration change.
        """
        self.gyro = gyro
        self.acc = acc
        self.mag = mag
        self.timestamp = timestamp
        self.alpha = alpha
        self.acc_change = acc_change


class RawReading:
    """ Represents a group of sensor readings from a hardware board containing multiple sensors (accelerometers,
    gyroscopes, and magnetometers) taken at a timestamp.

    The RawReading class provides methods to project the accelerometers and magnetometers to another location.

    Attributes:
        gyros (List[np.ndarray]): List of readings from gyroscope(s).
        accs (List[np.ndarray]): List of readings from accelerometer(s).
        mags (List[np.ndarray]): List of readings from magnetometer(s).
        timestamp (float): Timestamp of the reading.
    """
    gyros: List[np.ndarray]
    accs: List[np.ndarray]
    mags: List[np.ndarray]
    timestamp: float

    def __init__(self, gyros: List[np.ndarray], accs: List[np.ndarray], mags: List[np.ndarray], timestamp: float):
        """ Initializes a RawReading object with gyroscope, accelerometer, and magnetometer readings,
        and a timestamp."""
        self.gyros = gyros
        self.accs = accs
        self.mags = mags
        self.timestamp = timestamp

    def __str__(self):
        return f"RawReading(gyros={self.gyros}, accs={self.accs}, mags={self.mags}, timestamp={self.timestamp})"

    def __eq__(self, other: 'RawReading') -> bool:
        """ Compares two RawReading objects for equality. """
        if not isinstance(other, RawReading):
            return False
        elif len(self.gyros) != len(other.gyros):
            return False
        elif len(self.accs) != len(other.accs):
            return False
        elif len(self.mags) != len(other.mags):
            return False
        elif self.timestamp != other.timestamp:
            return False
        elif not all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(self.gyros, other.gyros)):
            return False
        elif not all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(self.accs, other.accs)):
            return False
        elif not all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(self.mags, other.mags)):
            return False
        else:
            return True

    def __add__(self, other: 'RawReading') -> 'RawReading':
        """ Adds two RawReading objects together. """
        assert np.isclose(self.timestamp, other.timestamp), "Timestamps must be equal to add RawReading objects."
        return RawReading(gyros=self.gyros + other.gyros, accs=self.accs + other.accs, mags=self.mags + other.mags,
                          timestamp=self.timestamp)

    def project_accs(self, alpha: np.ndarray, offsets: List[np.ndarray]):
        """ Projects the accelerometer readings to a point using rigid body dynamics with the given alpha
        and offsets from the sensor to the point.
        Args:
            alpha (np.ndarray): Angular acceleration of the board.
            offsets (List[np.ndarray]): Offsets from each accelerometer to the joint center. These offsets must be in
            the same order as the accelerometer readings.
        """
        assert len(offsets) == len(
            self.accs), f"The {len(offsets)} offsets were given, but {len(self.accs)} accelerometer readings are in this reading."
        assert len(set([tuple(offset) for offset in offsets])) == len(offsets), "Offsets must be unique."
        assert len(alpha) == 3, f"The alpha must be a 3D vector, but got {alpha}."
        gyro = np.mean(np.array(self.gyros), axis=0)
        projected_accs = [acc + np.cross(alpha, offset) + np.cross(gyro, np.cross(gyro, offset))
                          for acc, offset in zip(self.accs, offsets)]
        projected_acc = np.mean(np.array(projected_accs), axis=0)
        return projected_acc

    def project_mags(self, offsets: List[np.ndarray]):
        """ Projects the magnetometer readings to a point by linearly extrapolating multiple magnetometer readings.
        Args:
            offsets (List[np.ndarray]): Offsets from each magnetometer to the joint center. These offsets must be in
            the same order as the magnetometer readings.
        """
        assert len(offsets) == len(
            self.mags), (f"The {len(offsets)} offsets were given, but {len(self.mags)} magnetometer readings are in "
                         f"this reading.")
        assert len(set([tuple(offset) for offset in offsets])) == len(offsets), "Offsets must be unique."
        assert len(
            self.mags) > 1, f"At least two magnetometer readings are required to project the magnetometer reading but {len(self.mags)} are in this reading."
        projected_mags = []
        for i in range(len(self.mags) - 1):
            for j in range(i + 1, len(self.mags)):
                projected_mags.append(
                    self.projection_between_magnetometers(offsets[i], offsets[j], self.mags[i], self.mags[j]))
        projected_mag = np.mean(np.array(projected_mags), axis=0)
        return projected_mag

    @staticmethod
    def projection_between_magnetometers(mag_a_offset: np.ndarray, mag_b_offset: np.ndarray, mag_a: np.ndarray,
                                         mag_b: np.ndarray) -> np.ndarray:
        """ Projects a magnetometer reading between two magnetometers using linear interpolation.
        Args:
            mag_a_offset (np.ndarray): Offset from the first magnetometer to a point.
            mag_b_offset (np.ndarray): Offset from the second magnetometer to a point.
            mag_a (np.ndarray): Reading from the first magnetometer.
            mag_b (np.ndarray): Reading from the second magnetometer.
        """
        # Calculate projection for one pair
        mag_a_to_b = mag_a_offset - mag_b_offset
        imu_a_to_b_dist = np.linalg.norm(mag_a_to_b)
        mag_a_to_b_normalized = mag_a_to_b / imu_a_to_b_dist
        offset_along_a_to_b = np.dot(mag_a_offset, mag_a_to_b_normalized)
        percentage_along_a_to_b = offset_along_a_to_b / imu_a_to_b_dist
        weight_a, weight_b = (1 - percentage_along_a_to_b), percentage_along_a_to_b
        return (weight_a * mag_a) + (weight_b * mag_b)

    @staticmethod
    def create_readings_from_plate_trials(plate_trials: List[PlateTrial]) -> List[Dict[str, 'RawReading']]:
        readings = []
        for i in range(len(plate_trials[0])):
            reading_dict = {}
            for plate in plate_trials:
                reading_dict[plate.name] = RawReading(gyros=[plate.imu_trace.gyro[i], plate.second_imu_trace.gyro[i]],
                                                      accs=[plate.imu_trace.acc[i], plate.second_imu_trace.acc[i]],
                                                      mags=[plate.imu_trace.mag[i], plate.second_imu_trace.mag[i]],
                                                      timestamp=plate.imu_trace.timestamps[i])
            readings.append(reading_dict)
        return readings


class Board:
    """
    Represents a hardware board containing multiple sensors (accelerometers, gyroscopes, and magnetometers)
    placed at specific offsets relative to a joint center. The Board class provides methods to project
    sensor readings (stored in a RawReading) to a common joint center (stored in a ProjectedReading).

    To support the projection, the Board estimates the derivatives of the gyroscope readings and the acceleration
    readings by storing a history then using central finite differencing.

    Attributes:
        acc_to_joint_center_offsets (List[np.ndarray]): Offsets from each accelerometer to the joint center.
        mag_to_joint_center_offsets (List[np.ndarray]): Offsets from each magnetometer to the joint center.
        gyro_history (List[np.ndarray]): List to store recent gyroscope readings.
        acc_history (List[np.ndarray]): List to store recent accelerometer readings.
        timestamps (List[float]): List to store timestamps of recent readings.
        history_length (int): Number of past readings to retain for calculations.
    """
    acc_to_joint_center_offsets: List[np.ndarray]
    mag_to_joint_center_offsets: List[np.ndarray]
    gyro_history: List[np.ndarray]
    acc_history: List[np.ndarray]
    timestamps: List[float]
    history_length: int

    def __init__(self, acc_to_joint_center_offsets: List[np.ndarray], mag_to_joint_center_offsets: List[np.ndarray]):
        """
        Initializes a Board instance with offsets from accelerometers and magnetometers to the joint center.

        Args:
            acc_to_joint_center_offsets (List[np.ndarray]): List of offsets from each accelerometer to the joint center.
            mag_to_joint_center_offsets (List[np.ndarray]): List of offsets from each magnetometer to the joint center.
        """
        self.acc_to_joint_center_offsets = acc_to_joint_center_offsets
        self.mag_to_joint_center_offsets = mag_to_joint_center_offsets
        self.gyro_history = []
        self.acc_history = []
        self.timestamps = []
        self.history_length = 5

    def __eq__(self, other: 'Board'):
        """ Compares two Board objects for equality. """
        if not isinstance(other, Board):
            return False
        elif not all(np.array_equal(arr1, arr2) for arr1, arr2 in
                     zip(self.acc_to_joint_center_offsets, other.acc_to_joint_center_offsets)):
            return False
        elif not all(np.array_equal(arr1, arr2) for arr1, arr2 in
                     zip(self.mag_to_joint_center_offsets, other.mag_to_joint_center_offsets)):
            return False
        elif self.history_length != other.history_length:
            return False
        else:
            return True

    def to_dict(self) -> Dict[str, List[List[float]]]:
        """ Serializes the Board object into a dictionary with an "acc_to_joint_center_offsets" key and a
        "mag_to_joint_center_offsets" key. """
        return {
            "acc_to_joint_center_offsets": [offset.tolist() for offset in self.acc_to_joint_center_offsets],
            "mag_to_joint_center_offsets": [offset.tolist() for offset in self.mag_to_joint_center_offsets]
        }

    @staticmethod
    def from_dict(board_dict: Dict[str, List[List[float]]]) -> 'Board':
        """ Deserializes a dictionary with an "acc_to_joint_center_offsets" key and a "mag_to_joint_center_offsets" key
        into a Board object. """
        try:
            return Board(
                acc_to_joint_center_offsets=[np.array(offset) for offset in board_dict["acc_to_joint_center_offsets"]],
                mag_to_joint_center_offsets=[np.array(offset) for offset in board_dict["mag_to_joint_center_offsets"]])
        except KeyError as e:
            raise ValueError(f"Invalid board dictionary: {board_dict}. Missing key: {e}.")

    @staticmethod
    def from_board_config(board_center_offset: np.ndarray, board_center_to_acc_offsets: List[np.ndarray],
                          board_center_to_mag_offsets: List[np.ndarray]):
        """ Creates a Board object from a board configuration consisting of a board offset, and a list of sensor offsets
        relative to the board center. """
        acc_to_joint_center_offsets = [board_center_offset + offset for offset in board_center_to_acc_offsets]
        mag_to_joint_center_offsets = [board_center_offset + offset for offset in board_center_to_mag_offsets]
        return Board(acc_to_joint_center_offsets, mag_to_joint_center_offsets)

    def project_reading_to_joint_center(self, reading: RawReading, skip_acc: bool = False,
                                        skip_mag: bool = False,
                                        append_reading_to_history: bool = True) -> ProjectedReading:
        """ Projects a RawReading to the joint center, and returns a ProjectedReading.

        Args:
            reading (RawReading): The raw sensor reading.
            skip_acc (bool): If True, the accelerometer will be estimated by averaging.
            skip_mag (bool): If True, the magnetometer will be estimated by averaging.
            append_reading_to_history (bool): If True, the reading will be added to the history.
        Returns:
            ProjectedReading: The projected reading.
        """
        assert len(reading.gyros) > 0, "At least one gyroscope reading is required to project the reading."
        assert len(reading.accs) > 0, "At least one accelerometer reading is required to project the reading."
        assert len(reading.mags) > 0, "At least one magnetometer reading is required to project the reading."

        gyro = np.mean(reading.gyros, axis=0)
        if append_reading_to_history:
            self._update_timestamps_(timestamp=reading.timestamp)
            self._update_gyro_history_(latest_gyro=gyro)
        alpha = np.zeros(3) if skip_acc else self._estimate_alpha_()
        projected_acc = np.mean(reading.accs, axis=0) if skip_acc else reading.project_accs(alpha=alpha,
                                                                                            offsets=self.acc_to_joint_center_offsets)
        if append_reading_to_history:
            self._update_acc_history_(latest_acc=projected_acc)
        acc_change = self._estimate_acc_change_() if len(self.acc_history) > 1 else np.zeros(3)
        projected_mag = np.mean(reading.mags, axis=0) if skip_mag else reading.project_mags(
            offsets=self.mag_to_joint_center_offsets)
        return ProjectedReading(gyro, projected_acc, projected_mag, reading.timestamp, alpha, acc_change)

    def _estimate_alpha_(self) -> np.ndarray:
        """ This function estimates the angular acceleration of the board by taking the finite difference of the gyro
        history."""
        gyros = np.array(self.gyro_history)
        timestamps = np.array(self.timestamps)
        if len(gyros) < 2:
            return np.zeros(3)
        assert len(gyros) == len(
            timestamps), "The number of gyro readings must be equal to the number of timestamps to estimate alpha."
        alphas = np.array(
            [finite_difference(signal=gyros[:, i], timesteps=timestamps, method='central') for i in range(3)]).T
        alpha = np.mean(alphas, axis=0)
        return alpha

    def _estimate_acc_change_(self) -> np.ndarray:
        """
        This function estimates the change in acceleration, NOT THE RATE OF ACCELERATION, of the board by taking the
        finite difference of the acceleration history.
        """
        accs = np.array(self.acc_history)
        assert len(accs) > 1, "At least two acceleration readings are required to estimate the change in acceleration."
        delta_accs = np.array(
            [finite_difference(signal=accs[:, i], timesteps=np.arange(len(accs)), method='central') for i in
             range(3)]).T
        delta_acc = np.mean(delta_accs, axis=0)
        return delta_acc

    def _update_gyro_history_(self, latest_gyro: np.ndarray):
        """ Private method to update the gyro history. """
        self.gyro_history.append(latest_gyro)
        if len(self.gyro_history) > self.history_length:
            self.gyro_history.pop(0)

    def _update_acc_history_(self, latest_acc: np.ndarray):
        """ Private method to update the acceleration history. """
        self.acc_history.append(latest_acc)
        if len(self.acc_history) > self.history_length:
            self.acc_history.pop(0)

    def _update_timestamps_(self, timestamp: float):
        """ Private method to add a timestamp to the history and ensures history length does not exceed the limit. """
        self.timestamps.append(timestamp)
        if len(self.timestamps) > self.history_length:
            self.timestamps.pop(0)
