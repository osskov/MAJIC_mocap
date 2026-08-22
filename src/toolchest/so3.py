"""Vectorized SO(3) primitives shared by the joint-model fits and the filter constraints.

These lived in `experiments/joint_dof.py`, which fitted the joint models, until
`src/RelativeFilterPlus.py` needed the same formulas to USE those models as measurements.
`src/` cannot import from `experiments/` — the library does not depend on its consumers — so
they moved here and joint_dof imports them back under its own private names. There is still
exactly one implementation, and `test/TestJointDof.py` still guards it.

Everything is STACKED: inputs are (T, 3) rotation vectors or (T, 3, 3) matrices, outputs match.
A single-sample caller passes `v[None]` and takes `[0]`. That is slower than a scalar path would
be, and it is deliberate — the reference filter is the thing the compiled kernel is checked
against, so it must not carry a second implementation of these formulas that could drift.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v: np.ndarray) -> np.ndarray:
    """(T, 3) -> (T, 3, 3) stack of cross-product matrices."""
    zero = np.zeros(len(v))
    return np.stack([np.stack([zero, -v[:, 2], v[:, 1]], axis=-1),
                     np.stack([v[:, 2], zero, -v[:, 0]], axis=-1),
                     np.stack([-v[:, 1], v[:, 0], zero], axis=-1)], axis=-2)


def exp_matrix(v: np.ndarray) -> np.ndarray:
    """exp([v[t]]) for a (T, 3) stack of rotation vectors — Rodrigues, vectorized.

    Series-expanded below 1e-6 rad, where theta appears in a denominator. That is 6e-5 deg,
    far below anything reported, and the series is exact to machine precision there.
    """
    theta = np.linalg.norm(v, axis=1)
    K = skew(v)
    small = theta < 1e-6
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta ** 2 / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - np.cos(safe)) / safe ** 2)
    return np.eye(3) + a[:, None, None] * K + b[:, None, None] * (K @ K)


def right_jacobian(v: np.ndarray) -> np.ndarray:
    """The right Jacobian of SO(3) for a (T, 3) stack: d/dd exp([v + d]) = exp([v]) [J_r d].

    This is what turns a perturbation of the rotation VECTOR into a body-frame perturbation of
    the rotation, and it is why the curve model can be differentiated at all: the curve
    lives in the rotation vector, while the residual lives in the group.
    """
    theta = np.linalg.norm(v, axis=1)
    K = skew(v)
    small = theta < 1e-6
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - np.cos(safe)) / safe ** 2)
    b = np.where(small, 1.0 / 6.0 - theta ** 2 / 120.0, (safe - np.sin(safe)) / safe ** 3)
    return np.eye(3) - a[:, None, None] * K + b[:, None, None] * (K @ K)


def inverse_right_jacobian(v: np.ndarray) -> np.ndarray:
    """J_r(v)^-1 for a (T, 3) stack, in closed form.

        J_r^-1 = I + [v]/2 + (1/theta^2 - (1 + cos theta) / (2 theta sin theta)) [v]^2

    Closed form rather than `np.linalg.inv(right_jacobian(v))` because this is the piece the
    compiled kernel will have to carry, and a kernel cannot call a 3x3 inverse per constraint
    per step without allocating. Keeping the reference on the same formula means the two are
    compared on identical arithmetic rather than on two routes to the same matrix.

    The coefficient is singular at theta = 0 and again at theta = 2 pi. Only the first is
    reachable — a rotation vector returned by `log_matrix` has |v| <= pi — and it is
    series-expanded below 1e-6 rad, where the limit is 1/12.
    """
    theta = np.linalg.norm(v, axis=1)
    K = skew(v)
    small = theta < 1e-6
    safe = np.where(small, 1.0, theta)
    coefficient = np.where(
        small,
        1.0 / 12.0 + theta ** 2 / 720.0,
        1.0 / safe ** 2 - (1.0 + np.cos(safe)) / (2.0 * safe * np.sin(safe)))
    return np.eye(3) + 0.5 * K + coefficient[:, None, None] * (K @ K)


def inverse_left_jacobian(v: np.ndarray) -> np.ndarray:
    """J_l(v)^-1 = J_r(v)^-T, since J_l(v) = J_r(-v) = J_r(v)^T."""
    return np.swapaxes(inverse_right_jacobian(v), -1, -2)


def log_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation vectors of a (T, 3, 3) stack — the matrix logarithm, vectorized.

    Same result as Rotation.from_matrix(R).as_rotvec() to ~1e-13 but ~70x faster, which matters
    because this runs once per Gauss-Newton pass per outer evaluation.

    theta comes from arctan2(|w|, cos) rather than arccos so it stays accurate for small
    rotations, which is the regime the residuals converge into. The theta/sin(theta) scale does
    degrade as theta approaches pi, so those samples are handed to scipy's quaternion-based
    path instead; they only appear when a candidate fit is far off.
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = 0.5 * np.stack([R[:, 2, 1] - R[:, 1, 2],
                        R[:, 0, 2] - R[:, 2, 0],
                        R[:, 1, 0] - R[:, 0, 1]], axis=-1)
    sin_theta = np.linalg.norm(w, axis=1)
    theta = np.arctan2(sin_theta, np.clip(0.5 * (trace - 1.0), -1.0, 1.0))

    scale = np.where(sin_theta > 1e-8, theta / np.maximum(sin_theta, 1e-30), 1.0)
    rotvec = w * scale[:, None]

    near_pi = theta > 3.0
    if np.any(near_pi):
        rotvec[near_pi] = Rotation.from_matrix(R[near_pi]).as_rotvec()
    return rotvec


def transpose_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A^T B for stacks, formed directly rather than by transposing then multiplying."""
    return np.einsum('tji,tjk->tik', A, B)


def tangent_basis(axis: np.ndarray) -> np.ndarray:
    """A (3, 2) orthonormal basis for the plane perpendicular to `axis`.

    NOT canonical — which direction comes out first depends on the helper vector, so the two
    columns are 'some orthonormal pair', not 'adduction then rotation'. Any model that assigns
    meaning to the individual columns has to fit that assignment itself; see
    `fit_knee_coupling`, which searches the channel frame for exactly this reason.
    """
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return np.column_stack([e1, e2])
