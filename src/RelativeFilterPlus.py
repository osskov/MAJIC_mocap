from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation

class RelativeFilter:
    q_wp: Rotation
    q_wc: Rotation
    
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    num_sensors: int

    def __init__(self, gyro_std_parent: np.ndarray, gyro_std_child: np.ndarray, 
                 sensor_stds_parent: List[np.ndarray], sensor_stds_child: List[np.ndarray]):
        
        # --- Input Validation ---
        if gyro_std_parent.shape != (3,):
            raise ValueError("gyro_std_parent must be a NumPy array of shape (3,)")
        if gyro_std_child.shape != (3,):
            raise ValueError("gyro_std_child must be a NumPy array of shape (3,)")
        if len(sensor_stds_parent) != len(sensor_stds_child):
            raise ValueError("sensor_stds_parent and sensor_stds_child must have the same number of sensors.")
        if any(std.shape != (3,) for std in sensor_stds_parent) or any(std.shape != (3,) for std in sensor_stds_child):
            raise ValueError("Each standard deviation array in sensor_stds lists must have shape (3,).")
        
        self.num_sensors = len(sensor_stds_parent)
        if self.num_sensors == 0:
            raise ValueError("At least one sensor standard deviation must be provided.")

        # --- Matrix Initialization ---
        gyro_diag = np.concatenate([gyro_std_parent, gyro_std_child])
        self.Q = np.diag(gyro_diag**2)  # Use variance (std^2)
        
        # Interleave parent and child sensor stds for the R matrix
        sensor_diag = np.concatenate([s for pair in zip(sensor_stds_parent, sensor_stds_child) for s in pair])
        self.R = np.diag(sensor_diag**2)  # Use variance (std^2)
        
        self.P = np.eye(6)
        
        # Initialize rotations to identity
        self.q_wp = Rotation.identity()
        self.q_wc = Rotation.identity()

    def get_q_pc(self) -> Rotation:
        """Returns the relative rotation from parent to child."""
        return self.q_wp.inv() * self.q_wc

    def get_R_pc(self) -> np.ndarray:
        """Returns the relative rotation matrix from parent to child."""
        return self.get_q_pc().as_matrix()

    def update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, sensor_data_p: List[np.ndarray], sensor_data_c: List[np.ndarray], dt: float):
        """Performs a full prediction and measurement update cycle."""
        if len(sensor_data_p) != self.num_sensors or len(sensor_data_c) != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensor readings for parent and child, but got {len(sensor_data_p)} and {len(sensor_data_c)}.")

        q_lin_wp, q_lin_wc = self._get_time_update(gyro_p, gyro_c, dt)
        q_lin_wp, q_lin_wc = self._get_measurement_update(q_lin_wp, q_lin_wc, sensor_data_p, sensor_data_c)
        
        self.q_wp = q_lin_wp
        self.q_wc = q_lin_wc

    def set_qs(self, q_wp: Rotation, q_wc: Rotation):
        self.q_wp = q_wp
        self.q_wc = q_wc

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _get_time_update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, dt: float) -> Tuple[Rotation, Rotation]:
        """Predicts the next state based on gyroscope data."""
        R_p_update = Rotation.from_rotvec(dt * gyro_p).as_matrix()
        R_c_update = Rotation.from_rotvec(dt * gyro_c).as_matrix()
        
        F = np.zeros((6, 6))
        F[:3, :3] = R_p_update
        F[3:, 3:] = R_c_update
        
        G = np.eye(6) * dt

        q_lin_wp = self._get_gyro_orientation_estimate(self.q_wp, gyro_p, dt)
        q_lin_wc = self._get_gyro_orientation_estimate(self.q_wc, gyro_c, dt)

        # Predict covariance
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T
        return q_lin_wp, q_lin_wc

    def _get_gyro_orientation_estimate(self, q: Rotation, gyro: np.ndarray, dt: float) -> Rotation:
        """Integrates gyroscope readings to estimate orientation."""
        delta_q = Rotation.from_rotvec(dt * gyro)
        return q * delta_q

    def _get_measurement_update(self, q_lin_wp: Rotation, q_lin_wc: Rotation, sensor_data_p: List[np.ndarray], sensor_data_c: List[np.ndarray]) -> Tuple[Rotation, Rotation]:
        """Corrects the state prediction using sensor measurements."""
        normalized_sensor_data_p = [data / np.linalg.norm(data) if np.linalg.norm(data) != 0 else data for data in sensor_data_p]
        normalized_sensor_data_c = [data / np.linalg.norm(data) if np.linalg.norm(data) != 0 else data for data in sensor_data_c]

        R_wp = q_lin_wp.as_matrix()
        R_wc = q_lin_wc.as_matrix()

        H = self.get_H_jacobian(R_wp, R_wc, normalized_sensor_data_p, normalized_sensor_data_c)
        e = self.get_h(R_wp, R_wc, normalized_sensor_data_p, normalized_sensor_data_c)
        M = self.get_M_jacobian(R_wp, R_wc)
        
        # Kalman gain calculation
        S = H @ self.P @ H.T + M @ self.R @ M.T
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State and covariance update
        P_tilde = (np.eye(len(self.P)) - K @ H) @ self.P
        n = -K @ e
        
        # Apply correction to orientation
        q_delta_p = Rotation.from_rotvec(n[:3])
        q_delta_c = Rotation.from_rotvec(n[3:])
        
        q_lin_wp = q_lin_wp * q_delta_p
        q_lin_wc = q_lin_wc * q_delta_c
  
        # Update covariance matrix with the rotation correction
        J = np.eye(6)
        J[:3, :3] = Rotation.from_rotvec(n[:3]).as_matrix()
        J[3:, 3:] = Rotation.from_rotvec(n[3:]).as_matrix()
        self.P = J @ P_tilde @ J.T
        
        return q_lin_wp, q_lin_wc

    @staticmethod
    def get_H_jacobian(R_wp: np.ndarray, R_wc: np.ndarray, sensor_data_p: List[np.ndarray], sensor_data_c: List[np.ndarray]) -> np.ndarray:
        num_sensors = len(sensor_data_p)
        H = np.zeros((3 * num_sensors, 6))
        for i in range(num_sensors):
            H[3*i:3*(i+1), :3] = R_wp @ RelativeFilter.skew_symmetric(sensor_data_p[i]).T
            H[3*i:3*(i+1), 3:] = -R_wc @ RelativeFilter.skew_symmetric(sensor_data_c[i]).T
        return H

    @staticmethod
    def get_h(R_wp: np.ndarray, R_wc: np.ndarray, sensor_data_p: List[np.ndarray], sensor_data_c: List[np.ndarray], update: np.ndarray = np.zeros(6)) -> np.ndarray:
        num_sensors = len(sensor_data_p)
        h = np.zeros(3 * num_sensors)
        
        R_p_delta = Rotation.from_rotvec(update[:3]).as_matrix()
        R_c_delta = Rotation.from_rotvec(update[3:]).as_matrix()
        
        for i in range(num_sensors):
            p_world_frame = R_wp @ R_p_delta @ sensor_data_p[i]
            c_world_frame = R_wc @ R_c_delta @ sensor_data_c[i]
            h[3*i:3*(i+1)] = p_world_frame - c_world_frame
            
        return h

    def get_M_jacobian(self, R_wp: np.ndarray, R_wc: np.ndarray, update: np.ndarray = np.zeros(6)) -> np.ndarray:
        M = np.zeros((3 * self.num_sensors, 6 * self.num_sensors))
        for i in range(self.num_sensors):
            M[3*i:3*(i+1), 6*i:6*i+3] = R_wp @ (np.eye(3) + RelativeFilter.skew_symmetric(update[:3]))
            M[3*i:3*(i+1), 6*i+3:6*(i+1)] = -R_wc @ (np.eye(3) + RelativeFilter.skew_symmetric(update[3:]))
        return M