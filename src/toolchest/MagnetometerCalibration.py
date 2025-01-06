import numpy as np
from typing import List, Dict
from scipy.optimize import least_squares


class MagnetometerCalibration:
    radius: float
    center: np.ndarray
    scaling: List[float]
    axis: List[np.ndarray]
    overall_scale: float

    def __init__(self):
        self.radius = 1.0
        self.center = np.zeros(3)
        self.scaling = [0.0, 0.0, 0.0]
        self.axis = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        self.overall_scale = 1.0

    def to_dict(self) -> Dict:
        return {
            'radius': self.radius,
            'center': self.center.tolist(),
            'scaling': self.scaling,
            'axis': [axis.tolist() for axis in self.axis],
            'overall_scale': self.overall_scale
        }

    def from_dict(self, data: Dict):
        self.radius = data['radius']
        self.center = np.array(data['center'])
        self.scaling = data['scaling']
        self.axis = [np.array(axis) for axis in data['axis']]
        self.overall_scale = data['overall_scale']

    def initialize(self, observations: np.ndarray):
        self.center = np.mean(observations, axis=0)
        self.radius = np.mean(np.linalg.norm(observations - self.center, axis=1))
        self.scaling = [0.0, 0.0, 0.0]
        self.axis = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        self.overall_scale = 1.0

    def fit_ellipsoid(self, observations: np.ndarray):
        # Initial guess for the parameters
        self.initialize(observations)
        initial_guess = self.pack_array()
        # Perform least-squares fitting
        def residuals(params):
            self.unpack_array(params)
            errors = self.get_error(observations)
            constraints = np.array(self.get_constraints()) * 1e3
            return np.concatenate([errors, constraints])
        result = least_squares(residuals, initial_guess)
        self.unpack_array(result.x)

    def pack_array(self) -> np.ndarray:
        return np.concatenate([self.center, [self.radius], self.scaling, self.axis[0], self.axis[1], self.axis[2]])

    def unpack_array(self, array: np.ndarray):
        self.center = array[:3]
        self.radius = array[3]
        self.scaling = array[4:7].tolist()
        self.axis = [array[7:10], array[10:13], array[13:16]]

    def process(self, observations: np.ndarray):
        # Dimensions of observations is N x 3
        shifted = observations - self.center
        matrix = self.get_matrix()
        transformed = np.dot(shifted, matrix.T)
        transformed *= self.overall_scale
        return transformed

    def get_error(self, observations: np.ndarray):
        transformed = self.process(observations)
        return np.linalg.norm(transformed, axis=1) - self.radius

    def get_matrix(self) -> np.ndarray:
        sum = np.eye(3)
        for i in range(len(self.axis)):
            sum += np.outer(self.axis[i], self.axis[i]) * self.scaling[i]
        return sum

    def get_constraints(self) -> List[float]:
        constraints: List[float] = []
        for i in range(len(self.axis)):
            # Norms should be 1
            constraints.append(np.dot(self.axis[i], self.axis[i]) - 1)
        for i in range(len(self.axis)):
            for j in range(i + 1, len(self.axis)):
                # Orthogonality
                constraints.append(np.dot(self.axis[i], self.axis[j]))
        # Sum of scaling must be 0
        constraints.append(np.sum(self.scaling))
        return constraints

