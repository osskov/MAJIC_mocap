from typing import Tuple, List, Optional, Dict
import numpy as np
from scipy.spatial.transform import Rotation

class RelativeFilter:
    """
    An Extended Kalman Filter to estimate the relative orientation between two
    segments (parent and child).
    
    It uses:
    1. Gyroscope data for prediction.
    2. Any number of 1DOF vector sensors (e.g., accelerometers) for update.
    3. An optional, mutually exclusive joint constraint (either 1DOF or 2DOF).
    
    The state is the 6D error-state rotation vector eta = [eta_p, eta_c].
    """
    
    q_wp: Rotation
    q_wc: Rotation
    
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    num_vector_sensors: int
    
    # Joint Constraint parameters
    joint_type: Optional[str] = None
    joint_params: Dict = {}
    joint_noise_dim: int = 0


    def __init__(self, gyro_std_parent: np.ndarray, gyro_std_child: np.ndarray, 
                 vector_sensor_stds_parent: List[np.ndarray], 
                 vector_sensor_stds_child: List[np.ndarray],
                 joint_type: Optional[str] = None,
                 # 1DOF Joint Args (Section 3.1)
                 dof1_axis_parent: Optional[np.ndarray] = None,
                 dof1_axis_child: Optional[np.ndarray] = None,
                 dof1_std: Optional[np.ndarray] = None,
                 # 2DOF Joint Args (Section 3.2)
                 dof2_axis_parent: Optional[np.ndarray] = None, 
                 dof2_axis_child: Optional[np.ndarray] = None, 
                 dof2_angle_rad: float = np.pi/2.0, 
                 dof2_std: Optional[float] = None):
        """
        Initializes the filter matrices.
        
        Args:
            gyro_std_parent: (3,) array of gyroscope standard deviations (rad/s) for the parent.
            gyro_std_child: (3,) array of gyroscope standard deviations (rad/s) for the child.
            vector_sensor_stds_parent: List of (3,) arrays, one for each 1DOF vector
                                       sensor's std dev (e.g., accelerometer) on the parent body.
            vector_sensor_stds_child: List of (3,) arrays, one for each 1DOF vector
                                      sensor's std dev (e.g., accelerometer) on the child body.
            joint_type: Optional string, either '1dof' or '2dof'. Specifies the joint constraint.
            
            dof1_axis_parent: (3,) vector for 1DOF joint on parent (y^J in paper).
            dof1_axis_child: (3,) vector for 1DOF joint on child (y^K in paper).
            dof1_std: (3,) std dev of the 1DOF constraint (e_link in paper).
            
            dof2_axis_parent: (3,) axis vector for 2DOF constraint on parent (v^J in paper).
            dof2_axis_child: (3,) axis vector for 2DOF constraint on child (u^K in paper).
            dof2_angle_rad: The fixed angle (alpha) between the 2DOF axes.
            dof2_std: The std dev of the 2DOF scalar measurement (dot product).
        """
        
        # --- Input Validation ---
        if gyro_std_parent.shape != (3,):
            raise ValueError("gyro_std_parent must be a NumPy array of shape (3,)")
        if gyro_std_child.shape != (3,):
            raise ValueError("gyro_std_child must be a NumPy array of shape (3,)")
        if len(vector_sensor_stds_parent) != len(vector_sensor_stds_child):
            raise ValueError("vector_sensor_stds_parent and _child must have the same number of sensors.")
        if any(std.shape != (3,) for std in vector_sensor_stds_parent) or any(std.shape != (3,) for std in vector_sensor_stds_child):
            raise ValueError("Each std dev array in vector_sensor_stds lists must have shape (3,).")
        
        self.num_vector_sensors = len(vector_sensor_stds_parent)
        self.joint_type = joint_type
        self.joint_params = {}

        # --- Process Noise Matrix Q ---
        gyro_diag = np.concatenate([gyro_std_parent, gyro_std_child])
        self.Q = np.diag(gyro_diag**2)  # Use variance (std^2)
        
        # --- Measurement Noise Matrix R ---
        # 1. Variances for 1DOF *vector sensors* (e.g., accelerometers)
        sensor_diag_1dof = np.concatenate([s for pair in zip(vector_sensor_stds_parent, vector_sensor_stds_child) for s in pair])
        sensor_var_1dof = sensor_diag_1dof**2
        all_variances = [sensor_var_1dof]
        self.joint_noise_dim = 0
        
        # 2. Variance for *joint constraint*
        if self.joint_type == '1dof':
            if dof1_axis_parent is None or dof1_axis_child is None or dof1_std is None:
                raise ValueError("For '1dof' joint, must provide dof1_axis_parent, dof1_axis_child, and dof1_std.")
            if dof1_axis_parent.shape != (3,) or dof1_axis_child.shape != (3,) or dof1_std.shape != (3,):
                raise ValueError("1DOF joint parameters must all have shape (3,)")
            
            self.joint_params = {
                'y_j': dof1_axis_parent / np.linalg.norm(dof1_axis_parent),
                'y_k': dof1_axis_child / np.linalg.norm(dof1_axis_child)
            }
            # This constraint's noise is 3D, as per paper Eq 1 (e_link)
            # We assume the noise std is for the *link error*, not the sensor vectors
            all_variances.append(dof1_std**2)
            self.joint_noise_dim = 3
            
        elif self.joint_type == '2dof':
            if dof2_axis_parent is None or dof2_axis_child is None or dof2_std is None:
                raise ValueError("For '2dof' joint, must provide dof2_axis_parent, dof2_axis_child, and dof2_std.")
            if dof2_axis_parent.shape != (3,) or dof2_axis_child.shape != (3,):
                 raise ValueError("2DOF joint axes must have shape (3,)")
            
            self.joint_params = {
                'v_j': dof2_axis_parent / np.linalg.norm(dof2_axis_parent),
                'u_k': dof2_axis_child / np.linalg.norm(dof2_axis_child),
                'sin_alpha': np.sin(dof2_angle_rad)
            }
            # This constraint's noise is 1D (scalar)
            all_variances.append(np.array([dof2_std**2]))
            self.joint_noise_dim = 1
        
        elif self.joint_type is not None:
            raise ValueError(f"Unknown joint_type: '{self.joint_type}'. Must be '1dof' or '2dof'.")
        
        if self.num_vector_sensors == 0 and self.joint_type is None:
            raise ValueError("At least one vector sensor or a joint constraint must be provided.")
        
        self.R = np.diag(np.concatenate(all_variances))
        
        # --- Covariance and State Initialization ---
        self.P = np.eye(6)
        self.q_wp = Rotation.identity()
        self.q_wc = Rotation.identity()

    def get_q_pc(self) -> Rotation:
        return self.q_wp.inv() * self.q_wc

    def get_R_pc(self) -> np.ndarray:
        return self.get_q_pc().as_matrix()

    def update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, 
               vector_sensor_data_p: List[np.ndarray], 
               vector_sensor_data_c: List[np.ndarray], dt: float):
        """Performs a full prediction and measurement update cycle."""
        if len(vector_sensor_data_p) != self.num_vector_sensors or len(vector_sensor_data_c) != self.num_vector_sensors:
            raise ValueError(f"Expected {self.num_vector_sensors} vector sensor readings for parent and child.")

        q_lin_wp, q_lin_wc = self._get_time_update(gyro_p, gyro_c, dt)
        q_lin_wp, q_lin_wc = self._get_measurement_update(
            q_lin_wp, q_lin_wc, 
            vector_sensor_data_p, 
            vector_sensor_data_c
        )
        
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

    def _get_measurement_update(self, q_lin_wp: Rotation, q_lin_wc: Rotation, 
                                vector_sensor_data_p: List[np.ndarray], 
                                vector_sensor_data_c: List[np.ndarray]) -> Tuple[Rotation, Rotation]:
        """Corrects the state prediction using sensor measurements."""
        normalized_sensor_data_p = [data / np.linalg.norm(data) if np.linalg.norm(data) != 0 else data for data in vector_sensor_data_p]
        normalized_sensor_data_c = [data / np.linalg.norm(data) if np.linalg.norm(data) != 0 else data for data in vector_sensor_data_c]

        R_wp = q_lin_wp.as_matrix()
        R_wc = q_lin_wc.as_matrix()

        # Get Jacobians and residual, evaluated at eta = 0
        H = self.get_H_jacobian(R_wp, R_wc, normalized_sensor_data_p, normalized_sensor_data_c)
        e = self.get_h(R_wp, R_wc, normalized_sensor_data_p, normalized_sensor_data_c)
        M = self.get_M_jacobian(R_wp, R_wc)
        
        # Kalman gain calculation
        S = H @ self.P @ H.T + M @ self.R @ M.T
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State and covariance update
        P_tilde = (np.eye(len(self.P)) - K @ H) @ self.P
        n = -K @ e  # Error-state correction vector
        
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

    def get_H_jacobian(self, R_wp: np.ndarray, R_wc: np.ndarray, 
                       vector_sensor_data_p: List[np.ndarray], 
                       vector_sensor_data_c: List[np.ndarray]) -> np.ndarray:
        """Calculates the measurement Jacobian H = dh/d_eta."""
        all_H_blocks = []
        
        # 1. 1DOF Vector Sensor Jacobian (e.g., accelerometers)
        # This is (dh_vec / d_eta)
        for i in range(self.num_vector_sensors):
            # As per paper Eq 1, but with eta=0 as evaluation point
            H_vec_p = R_wp @ self.skew_symmetric(vector_sensor_data_p[i]).T
            H_vec_c = -R_wc @ self.skew_symmetric(vector_sensor_data_c[i]).T
            H_block = np.hstack([H_vec_p, H_vec_c])
            all_H_blocks.append(H_block)
        
        # 2. Joint Constraint Jacobian
        if self.joint_type == '1dof':
            # This is (dh_joint1 / d_eta)
            # Same form as 1DOF vector sensors
            y_j = self.joint_params['y_j']
            y_k = self.joint_params['y_k']
            H_joint_p = R_wp @ self.skew_symmetric(y_j).T
            H_joint_c = -R_wc @ self.skew_symmetric(y_k).T
            H_block = np.hstack([H_joint_p, H_joint_c])
            all_H_blocks.append(H_block)
            
        elif self.joint_type == '2dof':
            # This is (dh_joint2 / d_eta), from paper Eq 7 & 9
            v_j = self.joint_params['v_j']
            u_k = self.joint_params['u_k']
            
            # Evaluate at eta
            v_j_world = R_wp @ v_j
            u_k_world = R_wc @ u_k
            
            dh_d_eta_p = np.cross(u_k_world, v_j_world)
            dh_d_eta_c = np.cross(v_j_world, u_k_world)
            
            H_block = np.hstack([dh_d_eta_p, dh_d_eta_c]).reshape(1, 6)
            all_H_blocks.append(H_block)
        
        if not all_H_blocks:
            return np.empty((0, 6))
        return np.vstack(all_H_blocks)


    def get_h(self, R_wp: np.ndarray, R_wc: np.ndarray, 
              vector_sensor_data_p: List[np.ndarray], 
              vector_sensor_data_c: List[np.ndarray]) -> np.ndarray:
        """Calculates the measurement residual h(eta)."""
        
        # Apply error-state rotation
        all_h_blocks = []

        # 1. 1DOF Vector Sensor residuals (e.g., accelerometers)
        for i in range(self.num_vector_sensors):
            p_world_frame = R_wp @ vector_sensor_data_p[i]
            c_world_frame = R_wc @ vector_sensor_data_c[i]
            h_block = p_world_frame - c_world_frame
            all_h_blocks.append(h_block)
        
        # 2. Joint Constraint residual
        if self.joint_type == '1dof':
            # Paper Eq 1 (re-arranged)
            y_j = self.joint_params['y_j']
            y_k = self.joint_params['y_k']
            p_world_frame = R_wp @ y_j
            c_world_frame = R_wc @ y_k
            h_block = p_world_frame - c_world_frame
            all_h_blocks.append(h_block)
            
        elif self.joint_type == '2dof':
            # Paper Eq 4
            v_j = self.joint_params['v_j']
            u_k = self.joint_params['u_k']
            sin_alpha = self.joint_params['sin_alpha']

            v_j_world = R_wp @ v_j
            u_k_world = R_wc @ u_k

            h_scalar = v_j_world.dot(u_k_world) - sin_alpha
            all_h_blocks.append(np.array([h_scalar]))

        if not all_h_blocks:
            return np.empty((0,))
        return np.concatenate(all_h_blocks)

    def get_M_jacobian(self, R_wp: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
        """
        Calculates the block-diagonal noise Jacobian M = dh/d_nu.
        """
        
        # Apply error-state rotation (using 1st-order approx for simplicity)
        num_meas_vec = 3 * self.num_vector_sensors
        num_noise_vec = 6 * self.num_vector_sensors
        
        num_meas_joint = self.joint_noise_dim
        num_noise_joint = self.joint_noise_dim
        
        total_meas = num_meas_vec + num_meas_joint
        total_noise = num_noise_vec + num_noise_joint
        
        M = np.zeros((total_meas, total_noise))
        
        # --- Block 1: (dh_vec / d_nu_vec) ---
        # Top-left block
        if self.num_vector_sensors > 0:
            M_vec = np.zeros((num_meas_vec, num_noise_vec))
            for i in range(self.num_vector_sensors):
                M_vec[3*i:3*(i+1), 6*i:6*i+3] = R_wp
                M_vec[3*i:3*(i+1), 6*i+3:6*(i+1)] = -R_wc
            M[:num_meas_vec, :num_noise_vec] = M_vec

        # --- Block 2: (dh_joint / d_nu_joint) ---
        # Bottom-right block
        if self.joint_type == '1dof':
            # Assumes noise is e_link, an additive 3D vector noise
            M[num_meas_vec:, num_noise_vec:] = np.eye(3)
        elif self.joint_type == '2dof':
            # Assumes noise is additive 1D scalar noise
            M[num_meas_vec:, num_noise_vec:] = np.eye(1)
            
        # Off-diagonal blocks (dh_vec / d_nu_joint) and (dh_joint / d_nu_vec)
        # are zero, as their noises are independent.
        
        return M