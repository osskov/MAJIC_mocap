from typing import Tuple
import numpy as np
import nimblephysics as nimble

class RelativeFilter:
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    q_wp: nimble.math.Quaternion
    q_wc: nimble.math.Quaternion

    def __init__(self, acc_std: np.ndarray = np.ones(3) * 0.05,
                 gyro_std: np.ndarray = np.ones(3) * 0.01,
                 mag_std: np.ndarray = np.ones(3) * 0.05):
        gyro_diag = np.concatenate([gyro_std, gyro_std])
        self.Q = np.diag(gyro_diag)
        sensor_diag = np.concatenate([acc_std, acc_std, mag_std, mag_std])
        self.R = np.diag(sensor_diag)
        self.P = np.eye(6)
        self.q_wp = nimble.math.Quaternion([1., 0., 0., 0.])
        self.q_wc = nimble.math.Quaternion([1., 0., 0., 0.])

    def get_q_pc(self) -> nimble.math.Quaternion:
        return self.q_wp.conjugate().multiply(self.q_wc)

    def get_R_pc(self) -> np.ndarray:
        return self.get_q_pc().to_rotation_matrix()

    def update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, acc_jc_p: np.ndarray, acc_jc_c: np.ndarray, mag_p: np.ndarray, mag_c: np.ndarray, dt: float):
        # A) Time Update
        q_lin_wp, q_lin_wc = self._get_time_update(gyro_p, gyro_c, dt)

        # B) Measurement Update
        q_lin_wp, q_lin_wc = self._get_measurement_update(q_lin_wp, q_lin_wc, acc_jc_p, acc_jc_c, mag_p, mag_c)

        # Output (you may need to define these variables and arrays)
        self.q_wp = q_lin_wp
        self.q_wc = q_lin_wc

    def set_qs(self, q_wp: nimble.math.Quaternion, q_wc: nimble.math.Quaternion):
        self.q_wp = q_wp
        self.q_wc = q_wc

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _get_time_update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, dt: float) -> Tuple[nimble.math.Quaternion, nimble.math.Quaternion]:
        # Generate the state update matrix
        F = np.zeros((6, 6))
        F[:3, :3] = nimble.math.expMapRot(dt * gyro_p)
        F[3:, 3:] = nimble.math.expMapRot(dt * gyro_c)
        G = np.eye(6) * dt

        # Make linear approximation of the quaternion
        q_lin_wp = self._get_gyro_orientation_estimate(self.q_wp, gyro_p, dt)
        q_lin_wc = self._get_gyro_orientation_estimate(self.q_wc, gyro_c, dt)

        self.P = F @ self.P @ F.T + G @ self.Q @ G.T
        return q_lin_wp, q_lin_wc

    def _get_gyro_orientation_estimate(self, q: nimble.math.Quaternion, gyro: np.ndarray, dt: float) -> nimble.math.Quaternion:
        delta_q = nimble.math.expToQuat(dt * gyro)
        return q.multiply(delta_q)

    def _get_measurement_update(self,
                                q_lin_wp: nimble.math.Quaternion,
                                q_lin_wc: nimble.math.Quaternion,
                                acc_jc_p: np.ndarray, acc_jc_c: np.ndarray,
                                mag_jc_p: np.ndarray, mag_jc_c: np.ndarray) -> Tuple[nimble.math.Quaternion, nimble.math.Quaternion]:
        # Normalize accelerometers to isolate direction
        acc_jc_p = acc_jc_p / np.linalg.norm(acc_jc_p)
        acc_jc_c = acc_jc_c / np.linalg.norm(acc_jc_c)

        #Normalize magnetometers to isolate direction
        if np.linalg.norm(mag_jc_c) != 0 or np.linalg.norm(mag_jc_p) != 0:
            mag_jc_p = mag_jc_p / np.linalg.norm(mag_jc_p)
            mag_jc_c = mag_jc_c / np.linalg.norm(mag_jc_c)

        # Get rotation matrices from quaternions
        R_wp = q_lin_wp.to_rotation_matrix()
        R_wc = q_lin_wc.to_rotation_matrix()

        # Get the measurement jacobians
        H = self.get_H_jacobian(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)
        e = self.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)
        M = self.get_M_jacobian(R_wp, R_wc)
        R = self.R
        if np.linalg.norm(mag_jc_c) == 0 or np.linalg.norm(mag_jc_p) == 0:
            H = H[:3, :]
            e = e[:3]
            R = R[:6, :6]
            M = M[:3, :6]

        # Calculate the Kalman gain, where H accounts for the error associated with the update and R accounts for the
        # error associated with the measurement.
        S = H @ self.P @ H.T + M @ R @ M.T
        K = self.P @ H.T @ np.linalg.inv(S)

        P_tilde = self.P - K @ S @ K.T

        # Calculate the innovation
        n = -K @ e

        # Apply the innovation to the quaternions
        q_lin_wp = q_lin_wp.multiply(nimble.math.expToQuat(n[:3]))
        q_lin_wc = q_lin_wc.multiply(nimble.math.expToQuat(n[3:]))

        # Update the covariance matrix according to the innovation
        J = np.eye(6)
        J[:3, :3] = nimble.math.expMapRot(n[:3])
        J[3:, 3:] = nimble.math.expMapRot(n[3:])
        self.P = J @ P_tilde @ J.T
        return q_lin_wp, q_lin_wc

    @staticmethod
    def get_H_jacobian(R_wp: np.ndarray, R_wc: np.ndarray, acc_jc_p: np.ndarray, acc_jc_c: np.ndarray, mag_jc_p: np.ndarray, mag_jc_c: np.ndarray) -> np.ndarray:
        H = np.zeros((6, 6))
        H[:3, :3] = R_wp @ RelativeFilter.skew_symmetric(acc_jc_p).T
        H[:3, 3:] = - R_wc @ RelativeFilter.skew_symmetric(acc_jc_c).T
        H[3:, :3] = R_wp @ RelativeFilter.skew_symmetric(mag_jc_p).T
        H[3:, 3:] = - R_wc @ RelativeFilter.skew_symmetric(mag_jc_c).T
        return H

    @staticmethod
    def get_h(R_wp: np.ndarray, R_wc: np.ndarray, acc_jc_p: np.ndarray, acc_jc_c: np.ndarray, mag_jc_p: np.ndarray, mag_jc_c: np.ndarray, update: np.ndarray = np.zeros(6)) -> np.ndarray:
        h = np.zeros(6)
        acc_jc_p_world_frame = R_wp @ nimble.math.expMapRot(update[:3]) @ acc_jc_p
        acc_jc_c_world_frame = R_wc @ nimble.math.expMapRot(update[3:]) @ acc_jc_c
        mag_jc_p_world_frame = R_wp @ nimble.math.expMapRot(update[:3]) @ mag_jc_p
        mag_jc_c_world_frame = R_wc @ nimble.math.expMapRot(update[3:]) @ mag_jc_c
        h[:3] = acc_jc_p_world_frame - acc_jc_c_world_frame
        h[3:] = mag_jc_p_world_frame - mag_jc_c_world_frame
        return h

    def get_M_jacobian(self, R_wp, R_wc, update: np.ndarray = np.zeros(6)) -> np.ndarray:
        M = np.zeros((6, 12))
        M[:3, :3] = R_wp @ (np.eye(3) + RelativeFilter.skew_symmetric(update[:3]))
        M[:3, 3:6] = -R_wc @ (np.eye(3) + RelativeFilter.skew_symmetric(update[3:]))
        M[3:, 6:9] = R_wp @ (np.eye(3) + RelativeFilter.skew_symmetric(update[:3]))
        M[3:, 9:] = -R_wc @ (np.eye(3) + RelativeFilter.skew_symmetric(update[3:]))
        return M