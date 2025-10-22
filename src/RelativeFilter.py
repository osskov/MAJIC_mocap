from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation # New import for SciPy rotations

class RelativeFilter:
    # Use Rotation objects to hold the parent and child orientations
    q_wp: Rotation
    q_wc: Rotation
    
    # Type hints for NumPy arrays (used here for documentation clarity)
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray

    def __init__(self, acc_std: np.ndarray = np.ones(3) * 0.05,
                 gyro_std: np.ndarray = np.ones(3) * 0.01,
                 mag_std: np.ndarray = np.ones(3) * 0.05):
        gyro_diag = np.concatenate([gyro_std, gyro_std])
        self.Q = np.diag(gyro_diag)
        sensor_diag = np.concatenate([acc_std, acc_std, mag_std, mag_std])
        self.R = np.diag(sensor_diag)
        self.P = np.eye(6)
        
        # Initialize rotations to identity (w=1, x=0, y=0, z=0)
        self.q_wp = Rotation.identity()
        self.q_wc = Rotation.identity()

    def get_q_pc(self) -> Rotation:
        """Returns the relative rotation from parent to child (R_pc)."""
        return self.q_wp.inv() * self.q_wc

    def get_R_pc(self) -> np.ndarray:
        """Returns the relative rotation matrix from parent to child (R_pc)."""
        return self.get_q_pc().as_matrix()

    def update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, acc_jc_p: np.ndarray, acc_jc_c: np.ndarray, mag_p: np.ndarray, mag_c: np.ndarray, dt: float):
        # A) Time Update
        q_lin_wp, q_lin_wc = self._get_time_update(gyro_p, gyro_c, dt)

        # B) Measurement Update
        q_lin_wp, q_lin_wc = self._get_measurement_update(q_lin_wp, q_lin_wc, acc_jc_p, acc_jc_c, mag_p, mag_c)

        # Output
        self.q_wp = q_lin_wp
        self.q_wc = q_lin_wc

    def set_qs(self, q_wp: Rotation, q_wc: Rotation):
        self.q_wp = q_wp
        self.q_wc = q_wc

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _get_time_update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, dt: float) -> Tuple[Rotation, Rotation]:
        # Generate the state update matrix (F)
        R_p_update = Rotation.from_rotvec(dt * gyro_p).as_matrix()
        R_c_update = Rotation.from_rotvec(dt * gyro_c).as_matrix()        
        F = np.zeros((6, 6))
        F[:3, :3] = R_p_update
        F[3:, 3:] = R_c_update
        
        G = np.eye(6) * dt

        # Make linear approximation of the quaternion
        q_lin_wp = self._get_gyro_orientation_estimate(self.q_wp, gyro_p, dt)
        q_lin_wc = self._get_gyro_orientation_estimate(self.q_wc, gyro_c, dt)

        # Covariance Prediction
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T
        return q_lin_wp, q_lin_wc

    def _get_gyro_orientation_estimate(self, q: Rotation, gyro: np.ndarray, dt: float) -> Rotation:
        delta_q = Rotation.from_rotvec(dt * gyro)
        return q * delta_q

    def _get_measurement_update(self,
                                q_lin_wp: Rotation,
                                q_lin_wc: Rotation,
                                acc_jc_p: np.ndarray, acc_jc_c: np.ndarray,
                                mag_jc_p: np.ndarray, mag_jc_c: np.ndarray) -> Tuple[Rotation, Rotation]:
        # Normalize accelerometers to isolate direction
        acc_jc_p = acc_jc_p / np.linalg.norm(acc_jc_p) if np.linalg.norm(acc_jc_p) != 0 else acc_jc_p
        acc_jc_c = acc_jc_c / np.linalg.norm(acc_jc_c) if np.linalg.norm(acc_jc_c) != 0 else acc_jc_c

        #Normalize magnetometers to isolate direction
        if np.linalg.norm(mag_jc_c) != 0 or np.linalg.norm(mag_jc_p) != 0:
            mag_jc_p = mag_jc_p / np.linalg.norm(mag_jc_p) if np.linalg.norm(mag_jc_p) != 0 else mag_jc_p
            mag_jc_c = mag_jc_c / np.linalg.norm(mag_jc_c) if np.linalg.norm(mag_jc_c) != 0 else mag_jc_c

        # Get rotation matrices from quaternions
        R_wp = q_lin_wp.as_matrix()
        R_wc = q_lin_wc.as_matrix()

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
        q_delta_p = Rotation.from_rotvec(n[:3])
        q_delta_c = Rotation.from_rotvec(n[3:])
        
        # q.multiply(q_other) -> q @ q_other
        q_lin_wp = q_lin_wp * q_delta_p
        q_lin_wc = q_lin_wc * q_delta_c
  
        J = np.eye(6)
        J[:3, :3] = Rotation.from_rotvec(n[:3]).as_matrix()
        J[3:, 3:] = Rotation.from_rotvec(n[3:]).as_matrix()
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
        
        # R = expMapRot(v) -> Rotation.from_rotvec(v).as_matrix()
        R_p_delta = Rotation.from_rotvec(update[:3]).as_matrix()
        R_c_delta = Rotation.from_rotvec(update[3:]).as_matrix()
        
        # Transformation application: R_wp @ R_p_delta @ acc_jc_p
        acc_jc_p_world_frame = R_wp @ R_p_delta @ acc_jc_p
        acc_jc_c_world_frame = R_wc @ R_c_delta @ acc_jc_c
        mag_jc_p_world_frame = R_wp @ R_p_delta @ mag_jc_p
        mag_jc_c_world_frame = R_wc @ R_c_delta @ mag_jc_c
        
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