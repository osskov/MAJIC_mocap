import numpy as np
from typing import List
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation


def finite_difference_rotations(rotation_matrices: List[np.ndarray], timestamps: np.ndarray) -> np.ndarray:
    """
    Computes the rotation rate (angular velocity) from a list of rotation matrices over time using finite differencing.
    :param rotation_matrices: List of 3x3 rotation matrices. These are assumed to all be in the same static frame, such
    as the world frame. So they're all R_wb, where w is the world frame and b is the body frame.
    :param timestamps: Array of timestamps corresponding to the rotation matrices.
    :return: Array of angular velocity vectors.
    """
    if len(rotation_matrices) < 2:
        return np.zeros((max(1, len(rotation_matrices)), 3))
        
    R = np.array(rotation_matrices)
    R_prev_T = R[:-1].transpose((0, 2, 1))
    R_curr = R[1:]
    
    R_rel = np.matmul(R_prev_T, R_curr)
    dts = np.diff(timestamps)
    
    # Avoid divide by zero
    dts = np.where(dts == 0, 1e-9, dts)
    
    omegas = Rotation.from_matrix(R_rel).as_rotvec() / dts[:, None]
    
    angular_velocities = np.vstack([omegas, omegas[-1:]])
    return angular_velocities


def integrate_rotations(angular_velocities: List[np.ndarray], timestamps: np.ndarray,
                        initial_rotation: np.ndarray = np.eye(3)) -> List[np.ndarray]:
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
    :param R_rel: The rotation matrix.
    :param dt: The time step over which the rotation matrix was integrated.
    :return: The angular velocity vector.

    This doesn't care where R_rel came from, it can be either a left (world frame) or a right (body frame) rotation
    matrix.
    """
    return Rotation.from_matrix(R_rel).as_rotvec() / dt


def rotation_matrix_to_angular_velocity_python(R_rel: np.ndarray, dt: float) -> np.ndarray:
    """
    Converts a rotation matrix to an angular velocity vector using Rodrigues' rotation formula.
    :param R_rel: The rotation matrix.
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
    return Rotation.from_rotvec(omega * dt).as_matrix()


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

    # Vectorized sum of outer products: P.T @ C (shape: 3x3)
    X_pc = np.asarray(parent_vectors).T @ np.asarray(child_vectors)

    u, s, vh = np.linalg.svd(X_pc, full_matrices=True)
    scales = np.eye(3)
    matrix = u @ scales @ vh
    # Ensure the determinant is always positive
    if np.linalg.det(matrix) < 0:
        scales[2, 2] = -1
        matrix = u @ scales @ vh

    return matrix


def relative_rotvec(rotvec_a: np.ndarray, rotvec_b: np.ndarray) -> np.ndarray:
    """Rotation vector of a * b^-1, for (N, 3) arrays of rotation vectors.

    Equivalent to

        (Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).as_rotvec()

    to machine precision (~1e-15 rad), but ~9x faster: scipy's per-call overhead
    dominates when the arithmetic is this small, and this path constructs no Rotation
    objects. It matters because experiment_utils.compute_error_stats calls it on every
    sample of every method of every trial — tens of millions of rows, where it was 45%
    of the whole statistics pass.

    scipy's conventions are mirrored exactly so the two agree rather than merely agreeing
    closely: the |angle| <= 1e-3 Taylor branches in both from_rotvec and as_rotvec, and
    the w >= 0 canonicalisation that puts the returned angle in [0, pi].
    """
    def to_quat(rotvec):
        rotvec = np.asarray(rotvec, dtype=np.float64)
        angle = np.sqrt(np.einsum('ij,ij->i', rotvec, rotvec))
        small = angle <= 1e-3
        safe = np.where(small, 1.0, angle)
        a2 = angle * angle
        scale = np.where(small,
                         0.5 - a2 / 48.0 + a2 * a2 / 3840.0,
                         np.sin(safe / 2.0) / safe)
        quat = np.empty((len(rotvec), 4))
        quat[:, :3] = rotvec * scale[:, None]
        quat[:, 3] = np.cos(angle / 2.0)
        return quat

    p = to_quat(rotvec_a)
    q = to_quat(rotvec_b)
    q[:, :3] *= -1.0                       # inverse of a unit quaternion is its conjugate

    # scipy's _compose_quat, scalar-last
    out = np.empty_like(p)
    out[:, 0] = p[:, 3]*q[:, 0] + q[:, 3]*p[:, 0] + p[:, 1]*q[:, 2] - p[:, 2]*q[:, 1]
    out[:, 1] = p[:, 3]*q[:, 1] + q[:, 3]*p[:, 1] + p[:, 2]*q[:, 0] - p[:, 0]*q[:, 2]
    out[:, 2] = p[:, 3]*q[:, 2] + q[:, 3]*p[:, 2] + p[:, 0]*q[:, 1] - p[:, 1]*q[:, 0]
    out[:, 3] = p[:, 3]*q[:, 3] - p[:, 0]*q[:, 0] - p[:, 1]*q[:, 1] - p[:, 2]*q[:, 2]

    # as_rotvec: canonicalise to w >= 0 so the angle lands in [0, pi], then take the log
    np.negative(out, out=out, where=(out[:, 3] < 0)[:, None])
    vector_norm = np.sqrt(np.einsum('ij,ij->i', out[:, :3], out[:, :3]))
    angle = 2.0 * np.arctan2(vector_norm, out[:, 3])
    small = angle <= 1e-3
    a2 = angle * angle
    scale = np.where(small,
                     2.0 + a2 / 12.0 + 7.0 * a2 * a2 / 2880.0,
                     angle / np.sin(np.where(small, 1.0, angle) / 2.0))
    return out[:, :3] * scale[:, None]
