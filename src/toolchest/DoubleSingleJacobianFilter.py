import numpy as np
from .gyro_utils import angular_velocity_to_rotation_matrix


class DoubleSingleJacobianFilter:
    mag: np.ndarray
    jac: np.ndarray
    jac_col: int

    new_data_weight_mag: float
    new_data_weight_jac: float
    decay_unobserved_jac: float

    def __init__(self):
        self.mag = np.zeros(3)
        self.jac = np.zeros((3, 3))
        self.jac_col = 1
        self.new_data_weight_mag = 0.05
        self.new_data_weight_jac = 0.05
        self.decay_unobserved_jac = 1.0

    def update(self, gyro: np.ndarray, dt: float, mag_avg: np.ndarray, mag_grad: np.ndarray):
        """
        This updates the filter with a new observation.
        """

        rot = angular_velocity_to_rotation_matrix(gyro, dt)

        # Rotate our running estimates into the current frame
        self.mag = rot.T @ self.mag
        self.jac = rot.T @ self.jac @ rot

        # Take a geometric filter on the data
        self.mag = self.mag * (1 - self.new_data_weight_mag) + mag_avg * self.new_data_weight_mag
        self.jac[:, self.jac_col] = self.jac[:, self.jac_col] * (1 - self.new_data_weight_jac) + mag_grad * self.new_data_weight_jac
        # Ensure the jacobian is symmetric
        self.jac[self.jac_col, :] = self.jac[:, self.jac_col]

        for col in range(3):
            if col == self.jac_col:
                continue
            for row in range(3):
                if row == self.jac_col:
                    continue
                self.jac[row, col] *= self.decay_unobserved_jac

    def get_mag_estimate(self):
        return self.mag

    def get_grad_estimate(self):
        return self.jac[:, self.jac_col]
