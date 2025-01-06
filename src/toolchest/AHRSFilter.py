from typing import List, Dict, Callable, Any, Optional
import ahrs.filters as filt
import numpy as np
from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace
import nimblephysics as nimble

# List of all available filters and a list of their init call, gyro req, acc req, mag req, expected accelerometer at
# rest and expected reading from a magnetometer.
# NOTE: The relationship between gravity and the expected accelerometer reading is not always the same.
# Filters commented out did not pass unit tests.
ALL_FILTERS = {"Angular Rate": [filt.AngularRate, True, False, False, [0, 0, 0], [0, 0, 0]],
               # "AQUA": [filt.AQUA, False, True, True, [0, 0, 1], [1, 0, 1], # Empty initialization error
               # "Complementary": [filt.Complementary, True, True, None, [0, 0, 1], [0, -1, 1]], Need to identify the reference frame
               # "Davenport": [filt.Davenport, False, True, True, [0, 0, 1], [1, 1, 1]],
               "EKF": [filt.EKF, True, True, True, [0, 0, 0], [0, 0, 0]],  # Reference is set at initialization
               # "FAMC": [filt.FAMC, False, True, True],  # Empty initialization error
               # "FLAE": [filt.FLAE, False, True, True, [0, 0, 0], [0, 0, 0]],
               # "Fourati": [filt.Fourati, True, True, True, [0, 0, 0], [0, 0, 0]],
               # "FQA": [filt.FQA, False, True, None], # Empty initialization error
               # "Madgwick": [filt.Madgwick, True, True, None, [0, 0, 1], [1, 0, 1]],
               "Mahony": [filt.Mahony, True, True, None, [0, 0, 1], [-1, 0, 1]],
               #  "OLEQ": [filt.OLEQ, False, True, True, [0, 0, 0], [0, 0, 0]], # Different grav and mag definitions
               # "QUEST": [filt.QUEST, False, True, True, [0, 0, 0], [0, 0, 0]],  # Failing randominitialization test
               # "ROLEQ": [filt.ROLEQ, True, True, True, [0, 0, 0], [0, 0, 0]],
               # "SAAM": [filt.SAAM, False, True, True], # Needs NaN handling implementation
               # "Tilt": [filt.Tilt, False, True, None, [0, 0, 1], [1, 0, 1]]
               }


class AHRSFilter:
    """ This class is able to initialize, store and update several filters (most are implementations from the ahrs
    package)."""
    gyro_req: bool = False
    acc_req: bool = False
    mag_req: bool = False
    a_ref: np.array = np.array([0, 0, 0])
    m_ref: np.array = np.array([0, 0, 0])
    last_q: np.ndarray = np.array([1., 0., 0., 0.])
    filter: any
    R_wa: np.array = np.eye(3)
    q_world_to_ahrs: np.array = np.array([1., 0., 0., 0.])

    def __init__(self, select_filter: str, world_reference_mag: np.ndarray, world_reference_acc: np.ndarray = np.array([0, 1, 0])):
        """ This function currently assumes that gravity is in y for the world frame and in z for the AHRS frame."""
        if select_filter in ALL_FILTERS.keys():
            self.filter_type = select_filter
            filter_call, self.gyro_req, self.acc_req, self.mag_req, self.a_ref, self.m_ref = ALL_FILTERS[select_filter]
            self.filter = filter_call()
        else:
            raise ValueError("Warning! " + select_filter + " isn't a valid filter type.")

        # Try and adjust the references in the filter to match the expected gravity and magnetic field
        if self.set_filter_reference_mag(world_reference_mag) and self.set_filter_reference_acc(world_reference_acc):
            self.a_ref = world_reference_acc / np.linalg.norm(world_reference_acc)
            self.m_ref = world_reference_mag / np.linalg.norm(world_reference_mag)
            self.R_wa = np.eye(3)
            self.q_world_to_ahrs = nimble.math.expToQuat(nimble.math.logMap(self.R_wa))
        elif self.acc_req or self.mag_req:
            # Compute the angle between the world gravity and magnetic field vectors
            m_world = world_reference_mag / np.linalg.norm(world_reference_mag)
            g_world = world_reference_acc / np.linalg.norm(world_reference_acc)
            world_angle = np.arccos(np.dot(g_world, m_world))

            # Recompute the magnetic field relative to gravity in the AHRS frame so that the angle matches the world angle
            g_ahrs = self.a_ref
            g_ahrs = g_ahrs / np.linalg.norm(g_ahrs)
            m_ahrs = self.m_ref
            m_ahrs = m_ahrs / np.linalg.norm(m_ahrs)
            mutually_orthogonal = np.cross(g_ahrs, m_ahrs)
            m_ahrs = np.cross(mutually_orthogonal, g_ahrs)
            m_ahrs = m_ahrs / np.linalg.norm(m_ahrs)
            m_ahrs = m_ahrs * np.sin(world_angle) + g_ahrs * np.cos(world_angle)
            ahrs_angle = np.arccos(np.dot(g_ahrs, m_ahrs))


            W = np.outer(g_world, g_ahrs) + np.outer(m_world, m_ahrs)
            # Perform Singular Value Decomposition (SVD) on the cross-covariance matrix
            U, S, VT = np.linalg.svd(W)
            # Ensure proper orientation of U and VT (to prevent reflections)
            if np.linalg.det(U) * np.linalg.det(VT) < 0:
                U[:, 2] *= -1
            # Calculate the optimal rotation matrix R
            R_wa = U @ VT
            self.R_wa = R_wa
            self.q_world_to_ahrs = nimble.math.expToQuat(nimble.math.logMap(R_wa))
        else:
            self.R_wa = np.eye(3)
            self.q_world_to_ahrs = nimble.math.expToQuat(nimble.math.logMap(self.R_wa))

        self.last_q = nimble.math.Quaternion(self.R_wa.T).wxyz()
        initial_R = self.get_last_R()
        np.testing.assert_allclose(initial_R, np.eye(3), atol=1e-5, rtol=1e-5)

    @staticmethod
    def convert_to_world_orientations(imu_trace: IMUTrace, select_filter: str, expected_mag: np.ndarray, expected_acc: Optional[np.ndarray] = None) -> WorldTrace:
        """ This function takes an IMUTrace object and a filter type and returns a WorldTrace object with the
        orientations of the IMUTrace object converted to the world frame. The expected gravity is the gravity vector,
        NOT the expected gravity reading."""
        ahrs_filter = AHRSFilter(select_filter, expected_mag, expected_acc)
        positions: List[np.ndarray] = []
        rotations: List[np.ndarray] = []

        positions.append(np.zeros(3))
        rotations.append(ahrs_filter.get_last_R())

        for i in range(1, len(imu_trace.timestamps)):
            dt: float = imu_trace.timestamps[i] - imu_trace.timestamps[i - 1]
            ahrs_filter.update(dt, imu_trace.acc[i], imu_trace.gyro[i], imu_trace.mag[i])
            positions.append(np.zeros(3))
            rotations.append(ahrs_filter.get_last_R())
        return WorldTrace(imu_trace.timestamps, positions, rotations)

    def update(self, dt: float, acc: np.array, gyro: np.array, mag: np.array = None):
        if hasattr(self.filter, 'frequency'):
            self.filter.frequency = 1 / dt
        if hasattr(self.filter, 'Dt'):
            self.filter.Dt = dt
        if hasattr(self.filter, 'dt'):
            self.filter.dt = dt

        if self.mag_req is False:
            # Update when magnetometer data is not required
            self.set_last_q(self.filter.update(self.last_q, gyro, 'series'))
        else:
            if mag is None:
                # Handle cases when magnetometer data is required but not provided
                if self.mag_req is True:
                    print(f"Could not implement {self.filter_type} filter because there was no magnetometer data.")
                # Handle cases where magnetometer data is optional and not provided
                elif self.mag_req is None:
                    if self.filter_type in ['AQUA', 'Madgwick', 'Mahony']:
                        self.set_last_q(self.filter.updateIMU(self.last_q, gyr=gyro, acc=acc))
                    elif self.filter_type in ['Tilt', 'FQA']:
                        self.set_last_q(self.filter.estimate(acc=acc))
                    else:
                        self.set_last_q(self.filter.am_estimation(acc=acc))
            else:
                # Handle cases when magnetometer data is provided
                if self.filter_type in ['Davenport', 'FAMC', 'FLAE', 'FQA', 'OLEQ', 'QUEST', 'SAAM', 'Tilt']:
                    self.set_last_q(self.filter.estimate(acc=acc, mag=mag))
                elif self.filter_type in ['AQUA', 'Madgwick', 'Mahony']:
                    self.set_last_q(self.filter.updateMARG(q=self.last_q, gyr=gyro, acc=acc, mag=mag))
                elif self.filter_type in ['EKF', 'Fourati', 'ROLEQ', 'Complementary']:
                    self.set_last_q(self.filter.update(self.last_q, gyr=gyro, acc=acc, mag=mag))
                else:
                    self.set_last_q(self.filter.am_estimation(acc=acc, mag=mag))

    def get_last_q(self) -> np.array:
        return self.q_world_to_ahrs.multiply(nimble.math.Quaternion(self.last_q)).wxyz()

    def get_last_R(self) -> np.ndarray:
        return self.R_wa @ nimble.math.Quaternion(self.last_q).to_rotation_matrix()

    def set_last_q(self, q: np.array):
        if hasattr(self.filter, 'q0'):
            self.filter.q0 = q.copy()
        self.last_q = q.copy()

    def get_qs_for_dataset(self, dt: np.array, gyro: np.array, acc: np.array, mag: np.array = None) -> np.array:
        qs = np.zeros((np.shape(acc)[0] + 1, 4))
        for i in range(np.shape(acc)[0] - 1):
            qs[i, :] = self.last_q
            if mag is not None:
                self.update(dt, acc[i, :], gyro[i, :], mag[i, :])
            else:
                self.update(dt, acc[i, :], gyro[i, :])
        return qs

    def set_filter_reference_acc(self, a_ref: np.array) -> bool:
        a_ref = a_ref / np.linalg.norm(a_ref)
        if hasattr(self.filter, 'a_ref'):
            self.filter.a_ref = a_ref
            return True
        elif hasattr(self.filter, 'g_q'):
            self.filter.g_q = a_ref
            return True
        elif hasattr(self.filter, 'ref'):
            original_ref = self.filter.ref
            self.filter.ref = np.vstack((a_ref, original_ref[1, :]))
            return True
        return False

    def set_filter_reference_mag(self, m_ref: np.array) -> bool:
        m_ref = m_ref / np.linalg.norm(m_ref)
        if hasattr(self.filter, 'm_ref'):
            self.filter.m_ref = m_ref
            return True
        elif hasattr(self.filter, 'm_q'):
            self.filter.m_q = m_ref
            return True
        elif hasattr(self.filter, 'ref'):
            original_ref = self.filter.ref
            self.filter.ref = np.vstack((original_ref[0, :], m_ref))
            return True
        return False
