import numpy as np
from typing import List, Union, Tuple
from scipy.linalg import logm, expm
import nimblephysics as nimble


def finite_difference_rotations(rotation_matrices: List[np.ndarray], timestamps: np.ndarray) -> List[np.ndarray]:
    """
    Computes the rotation rate (angular velocity) from a list of rotation matrices over time using finite differencing.
    :param rotation_matrices: List of 3x3 rotation matrices. These are assumed to all be in the same static frame, such
    as the world frame. So they're all R_wb, where w is the world frame and b is the body frame.
    :param timestamps: Array of timestamps corresponding to the rotation matrices.
    :return: List of angular velocity vectors.
    """
    angular_velocities = []
    for i in range(1, len(rotation_matrices)):
        # Relative rotation matrix. This assumes that R_rel is in the body frame, and is right multiplied by the
        # previous rotation matrix to get the current rotation matrix.
        #
        # R_current = R_previous * R_rel
        #
        # Therefore:
        #
        # R_rel = R_previous.T * R_current
        R_rel = np.dot(rotation_matrices[i - 1].T, rotation_matrices[i])

        # Compute the time difference
        dt: float = timestamps[i] - timestamps[i - 1]

        # Compute the angular velocity
        omega = rotation_matrix_to_angular_velocity(R_rel, dt)

        assert not np.isnan(omega).any(), f"NaN in omega at index {i}"

        angular_velocities.append(omega)

    # Extend the angular velocities to the same length as the rotation matrices
    if len(angular_velocities) == 0:
        return [np.zeros(3)]
    angular_velocities.append(angular_velocities[-1])

    return angular_velocities


def integrate_rotations(angular_velocities: List[np.ndarray], timestamps: np.ndarray, initial_rotation: np.ndarray = np.eye(3)) -> List[np.ndarray]:
    """
    Integrate a list of angular velocities to get a list of rotation matrices.
    :param angular_velocities: List of angular velocities.
    :param timestamps: Array of timestamps corresponding to the angular velocities.
    :param initial_rotation: The initial rotation matrix, defaults to identity (I).
    :return: List of rotation matrices.
    """
    R: np.ndarray = initial_rotation
    rotation_matrices = [R]
    for i in range(1, len(angular_velocities)):
        dt: float = timestamps[i] - timestamps[i - 1]
        R = np.dot(R, angular_velocity_to_rotation_matrix(angular_velocities[i], dt))
        rotation_matrices.append(R)
    return rotation_matrices


def rotation_matrix_to_angular_velocity(R_rel: np.ndarray, dt: float) -> np.ndarray:
    """
    Converts a rotation matrix to an angular velocity vector using Rodrigues' rotation formula.
    :param R_rel The rotation matrix.
    :param dt: The time step over which the rotation matrix was integrated.
    :return: The angular velocity vector.

    This doesn't care where R_rel came from, it can be either a left (world frame) or a right (body frame) rotation
    matrix.
    """
    return nimble.math.logMap(R_rel) / dt

def rotation_matrix_to_angular_velocity_python(R_rel: np.ndarray, dt: float) -> np.ndarray:
    """
    Converts a rotation matrix to an angular velocity vector using Rodrigues' rotation formula.
    :param R_rel The rotation matrix.
    :param dt: The time step over which the rotation matrix was integrated.
    :return: The angular velocity vector.

    This doesn't care where R_rel came from, it can be either a left (world frame) or a right (body frame) rotation
    matrix.
    """
    # Compute the matrix logarithm of the relative rotation, in the world frame
    log_R_rel: np.ndarray = logm(R_rel)

    # Extract the skew-symmetric matrix (angular velocity matrix)
    omega_matrix = log_R_rel / dt

    # Extract the angular velocity vector
    omega = np.array([
        omega_matrix[2, 1],
        omega_matrix[0, 2],
        omega_matrix[1, 0]
    ])

    return omega


def angular_velocity_to_rotation_matrix(omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Converts an angular velocity vector to a rotation matrix using Rodrigues' rotation formula.
    :param omega: The angular velocity vector.
    :param dt: The time step over which the angular velocity is integrated.
    :return: The relative rotation matrix, R_rel.

    This doesn't care where R_rel came from, it can be either a left (world frame) or a right (body frame) rotation
    matrix.
    """
    return nimble.math.expMapRot(omega * dt)

def angular_velocity_to_rotation_matrix_python(omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Converts an angular velocity vector to a rotation matrix using Rodrigues' rotation formula.
    :param omega: The angular velocity vector.
    :param dt: The time step over which the angular velocity is integrated.
    :return: The relative rotation matrix, R_rel.

    This doesn't care where R_rel came from, it can be either a left (world frame) or a right (body frame) rotation
    matrix.
    """
    theta = np.linalg.norm(omega) * dt
    if theta == 0:
        return np.eye(3)

    omega_hat = omega / np.linalg.norm(omega)
    K = np.array([
        [0, -omega_hat[2], omega_hat[1]],
        [omega_hat[2], 0, -omega_hat[0]],
        [-omega_hat[1], omega_hat[0], 0]
    ])

    # You can left-multiply an existing world rotation by this matrix to rotate it by the angular velocity
    R_delta = expm(theta * K)

    return R_delta


def calculate_best_fit_rotation(parent_vectors: List[np.ndarray], child_vectors: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the best fit rotation matrix between two sets of vectors using the Wahba problem.
    :param parent_vectors: List of 3D vectors in the parent frame.
    :param child_vectors: List of 3D vectors in the child frame.
    :return: The best fit rotation matrix from the parent frame to the child frame (R_parent_child).
    """
    if len(parent_vectors) != len(child_vectors):
        raise ValueError("Parent and child vectors must be the same length.")

    X_pc = np.zeros((3, 3))
    for parent, child in zip(parent_vectors, child_vectors):
        X_pc += np.outer(parent, child)

    u, s, vh = np.linalg.svd(X_pc, full_matrices=True)
    scales = np.eye(3)
    matrix = u @ scales @ vh
    # Ensure the determinant is always positive
    if np.linalg.det(matrix) < 0:
        scales[2, 2] = -1
        matrix = u @ scales @ vh

    return matrix