"""Joint models expressed so a filter can use them as measurements.

`experiments/joint_dof.py` FITS these models to reference kinematics and reports how wrong each
one is. This module is the other half: the same models, in the form
`src/RelativeFilterPlus.py` needs to feed them to the measurement update. The published curve
and the SO(3) primitives live below `src/` so both halves share one implementation — see
`src/toolchest/so3.py` for the same argument about the group operations.

Four models, in the order of the ladder joint_dof measures:

    hinge      one axis matched in both frames.       3 residual dims, rank 2.
    coupling   the published 1-DOF nonlinear knee.    2 residual dims.
    universal  the angle between two axes is fixed.   1 residual dim.
    spring     one channel pulled toward neutral.     1 residual dim, deliberately weak.

The hinge and universal residuals are algebraic in the two orientations and are implemented
directly in the filter. The coupling and spring residuals both need the relative rotation's
logarithm in the model's own neutral frame, which is what `relative_rotvec_in_model_frame`
and `model_frame_jacobian` provide.

CONVENTION, and it is the one joint_dof fits to: the model is

    R_pc(t) = R_0 exp([nu(t)]),        nu in the CHILD frame,

so `nu = log(R_0^T R_pc)` and every axis this module takes (`u_c` for the coupling, the neutral
channel for the spring) is a direction in the child frame. `joint_dof.fit_knee_coupling` stores
exactly this: `axis_child` is u_c and `axis_parent` is R_0 @ u_c.
"""
from typing import Tuple

import numpy as np

from src.toolchest import so3

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

# Reuben et al. (1986) knee coupling, transcribed from IMoveLab's
# mocap_ref/utils/mt/sfa_cf.py::correct_nonsagittal_knee. Argument and result are both in
# DEGREES, and both polynomials have no constant term — the coupling is zero at zero flexion
# by construction, which is what makes the flexion zero identifiable. Adduction is the
# quartic, internal rotation the cubic.
REUBEN_ADDUCTION_COEFFS = np.array([0.0791, -5.733e-4, -7.682e-6, 5.759e-8])
REUBEN_ROTATION_COEFFS = np.array([0.3695, -2.958e-3, 7.666e-6, 0.0])


def reuben_design(q: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """(c, dc/dq) of the two PUBLISHED channels at joint angle q — evaluated exactly.

    Exactly, rather than by least-squares projection onto a spline basis, which is how the
    first version reached it so that one code path could serve both curves. That projection was
    not free: a quartic with Reuben's coefficients over 85 deg of flexion is not in the span of
    a cubic B-spline with a handful of interior knots, and the leftover showed up as 0.84 deg
    RMS on a SYNTHETIC knee built to obey the coupling exactly — a joint whose true residual is
    zero. That is the same order as the differences this resolves, and it was being charged to
    the published model.

    Flexion is CLIPPED at zero. The polynomials are zero at zero by construction and carry large
    negative coefficients, so at negative flexion they diverge rather than extrapolate — the
    rotation channel reads +13 deg at 60 deg of flexion and -84 deg at -108. Inside the clip the
    curve is held at its value at zero, so its slope contributes nothing to dnu/dq there.

    IN THE FILTER the clip matters for a second reason joint_dof never faced. joint_dof
    evaluates the curve at a flexion angle solved from the reference; the filter evaluates it
    at a flexion angle read off its own current ESTIMATE, which early in a trial can be far
    from the truth and on the wrong side of zero. Without the clip a transient extension error
    would inject an 84 deg target into the measurement update and the filter would chase it.

    Units: the polynomials are quoted in degrees of output per degree of input, so x is in
    degrees and the result is converted to radians. dx/dq contributes sigma * RAD2DEG and the
    radian conversion contributes DEG2RAD, which cancel to a bare sigma.
    """
    flexion = sigma * q
    x = np.clip(flexion, 0.0, None) * RAD2DEG
    powers = np.stack([x, x ** 2, x ** 3, x ** 4], axis=-1)
    slopes = np.stack([np.ones_like(x), 2 * x, 3 * x ** 2, 4 * x ** 3], axis=-1)
    channels = np.stack([REUBEN_ADDUCTION_COEFFS, REUBEN_ROTATION_COEFFS], axis=-1)  # (4, 2)
    derivative = (slopes @ channels) * sigma
    derivative[flexion < 0.0] = 0.0
    return (powers @ channels) * DEG2RAD, derivative


def relative_rotvec_in_model_frame(R_wp: np.ndarray, R_wc: np.ndarray,
                                   R_0: np.ndarray) -> np.ndarray:
    """nu = log(R_0^T R_pc) for one sample, the coordinate both soft models are written in.

    nu = 0 is the model's neutral pose. Its component along the joint axis is the joint angle
    (that is the gauge joint_dof fits in), and its components across the axis are the two
    channels the coupling pins to Reuben's curve and the spring pulls toward zero.
    """
    R_pc = R_wp.T @ R_wc
    return so3.log_matrix((R_0.T @ R_pc)[None])[0]


def model_frame_jacobian(nu: np.ndarray, R_0: np.ndarray) -> np.ndarray:
    """d(nu) / d(eta) for the 6-D error state eta = [eta_p, eta_c], as a (3, 6) block.

    The error state perturbs each orientation on the right, R_wp -> R_wp exp([eta_p]) and
    R_wc -> R_wc exp([eta_c]), so

        R_0^T R_pc  ->  R_0^T exp([-eta_p]) R_wp^T R_wc exp([eta_c])
                     =  exp([-R_0^T eta_p]) exp([nu]) exp([eta_c]),

    a LEFT perturbation by -R_0^T eta_p and a RIGHT perturbation by eta_c. Composing
    exp([a]) exp([nu]) exp([b]) and reading off the rotation vector to first order gives

        d(nu) = -J_l(nu)^-1 R_0^T eta_p  +  J_r(nu)^-1 eta_c.

    The asymmetry between the two blocks is real and is not a sign convention: a parent
    perturbation acts on the far side of the accumulated rotation from a child one, so they are
    carried by different Jacobians and coincide only at nu = 0.
    """
    inverse_left = so3.inverse_left_jacobian(nu[None])[0]
    inverse_right = so3.inverse_right_jacobian(nu[None])[0]
    return np.hstack([-inverse_left @ R_0.T, inverse_right])


def spring_std_for_gain(gain: float, prior_variance: float) -> float:
    """The measurement std that reproduces IMoveLab's feedback gain kappa, for comparison.

    Their neutral-angle term is not a measurement update: it applies a fixed fraction kappa of
    the deviation as a correction after the filter step. A scalar Kalman update with prior
    variance p and measurement variance s^2 applies the fraction p / (p + s^2), so matching the
    two gives

        s^2 = p (1 - kappa) / kappa.

    At their published gains this is weak by construction: kappa = 0.0064 (hip) is s^2 ~ 155 p,
    i.e. the constraint barely moves the estimate, and kappa = 0.25 (ankle) is s^2 = 3 p.

    The mapping is exact only instantaneously — p is the filter's own prior variance along the
    constrained direction and moves every step, whereas kappa is fixed — so this converts a
    gain into a COMPARABLE std at a stated operating point, not into an equivalent filter.
    That is enough for its purpose, which is to run our arm at their strength rather than at a
    strength we chose.
    """
    if not 0.0 < gain <= 1.0:
        raise ValueError(f"gain must be in (0, 1], got {gain}")
    if prior_variance <= 0.0:
        raise ValueError(f"prior_variance must be positive, got {prior_variance}")
    return float(np.sqrt(prior_variance * (1.0 - gain) / gain))
