"""Compiled batch implementation of RelativeFilter's update loop.

src/RelativeFilterPlus.py stays the reference implementation: it is what the paper's
equations are annotated against, what test/TestRelativeFilter.py exercises, and what
supports the 1DOF/2DOF joint constraints. This module is an optimisation of the one
configuration the experiment pipeline actually runs -- two 3-axis vector sensors
(accelerometer, magnetometer) and no joint constraint -- and nothing else.

test/TestRelativeFilterFast.py is the contract between the two. If the reference
changes, that test fails until this is brought back into line.

Why a whole-loop kernel rather than a compiled update():
    The filter is a sequential recursion, so there is nothing to parallelise. The
    cost being removed is per-step interpreter and dispatch overhead -- roughly 30
    numpy calls per step, each allocating, on 6x6 arrays whose arithmetic is a few
    hundred flops. Compiling one step and calling it 60,000 times from Python would
    pay numba's boundary crossing every step and give most of that back, so the loop
    lives inside the kernel and the boundary is crossed once.

Numerics: the state is carried as scipy-convention quaternions (scalar-last, compose
then renormalise, the |angle| <= 1e-3 Taylor branch in from_rotvec) so that the result
tracks the reference to ~1e-6 degrees over a 60,000-step trial rather than drifting.
Not bit-for-bit: the 6x6 products accumulate in a different order.
"""
from typing import List, Optional

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:                                              # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op stand-in so the module still imports without numba.

        The kernel is written as explicit scalar loops, so running it uncompiled would
        be far SLOWER than the reference class. Callers must branch on
        NUMBA_AVAILABLE rather than falling through to here.
        """
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


NUM_VECTOR_SENSORS = 2          # accelerometer, magnetometer
MEAS_DIM = 3 * NUM_VECTOR_SENSORS


# ---------------------------------------------------------------------------
# quaternion helpers, mirroring scipy.spatial.transform.Rotation exactly
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def _rotvec_to_quat(rx, ry, rz, out):
    angle = np.sqrt(rx * rx + ry * ry + rz * rz)
    if angle <= 1e-3:
        a2 = angle * angle
        scale = 0.5 - a2 / 48.0 + a2 * a2 / 3840.0
    else:
        scale = np.sin(angle / 2.0) / angle
    out[0] = rx * scale
    out[1] = ry * scale
    out[2] = rz * scale
    out[3] = np.cos(angle / 2.0)


@njit(cache=True, inline='always')
def _quat_mul(p, q, out):
    """scipy's _compose_quat followed by the renormalisation __mul__ applies."""
    o0 = p[3] * q[0] + q[3] * p[0] + p[1] * q[2] - p[2] * q[1]
    o1 = p[3] * q[1] + q[3] * p[1] + p[2] * q[0] - p[0] * q[2]
    o2 = p[3] * q[2] + q[3] * p[2] + p[0] * q[1] - p[1] * q[0]
    o3 = p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2]
    nrm = np.sqrt(o0 * o0 + o1 * o1 + o2 * o2 + o3 * o3)
    out[0] = o0 / nrm
    out[1] = o1 / nrm
    out[2] = o2 / nrm
    out[3] = o3 / nrm


@njit(cache=True, inline='always')
def _quat_to_mat(q, m):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w
    m[0, 0] = x2 - y2 - z2 + w2
    m[0, 1] = 2.0 * (xy - zw)
    m[0, 2] = 2.0 * (xz + yw)
    m[1, 0] = 2.0 * (xy + zw)
    m[1, 1] = -x2 + y2 - z2 + w2
    m[1, 2] = 2.0 * (yz - xw)
    m[2, 0] = 2.0 * (xz - yw)
    m[2, 1] = 2.0 * (yz + xw)
    m[2, 2] = -x2 - y2 + z2 + w2


@njit(cache=True, inline='always')
def _congruence(A, B, P, r0, c0, out):
    """out = A @ P[r0:r0+3, c0:c0+3] @ B.T"""
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                v = 0.0
                for m in range(3):
                    v += A[i, m] * P[r0 + m, c0 + k]
                s += v * B[j, k]
            out[i, j] = s


@njit(cache=True, inline='always')
def _solve_spd(S, rhs, L, out):
    """out[c, :] solves S @ x = rhs[c, :], for each of the rhs.shape[0] right-hand
    sides, by Cholesky factorisation of S.

    Used to form K = P H^T inv(S) without inverting: since S is symmetric,
    K[i, :] is the solution of S x = (P H^T)[i, :]. That is ~8x cheaper than a 6x6
    np.linalg.inv, which is otherwise about two thirds of a step.

    S = H P H^T + M R M^T is SPD by construction, but only the lower triangle is
    read, so a P that has drifted from exact symmetry is implicitly symmetrised.
    Returns False on a non-positive pivot so the caller can fall back to an explicit
    inverse rather than propagate a NaN through the rest of a 60,000-sample trial:
    the covariance update uses the plain (I - K H) P form rather than Joseph, so P
    is not guaranteed symmetric over a long run.
    """
    n = S.shape[0]
    for i in range(n):
        for j in range(i + 1):
            s = S[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    return False
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    for c in range(rhs.shape[0]):
        for i in range(n):                       # forward: L y = b
            s = rhs[c, i]
            for k in range(i):
                s -= L[i, k] * out[c, k]
            out[c, i] = s / L[i, i]
        for i in range(n - 1, -1, -1):           # back: L^T x = y
            s = out[c, i]
            for k in range(i + 1, n):
                s -= L[k, i] * out[c, k]
            out[c, i] = s / L[i, i]
    return True


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

@njit(cache=True)
def _kernel(gyro_p, gyro_c, vp, vc, dt, Q, var_p, var_c,
            q_wp0, q_wc0, P0, normalize):
    """vp/vc are (N, 2, 3): [accelerometer, magnetometer] in each body frame.

    A sensor is switched off for a sample by zeroing its reading, which is how
    mag_mode 'off' and 'adapt' already work. That makes its rows of H exactly zero;
    because M R M^T is block diagonal across sensors (each draws on a disjoint slice
    of the noise vector), S stays block diagonal AND stays invertible on the strength
    of the surviving noise block, so the dead sensor's columns of K are exactly zero
    and it contributes nothing to either the state or the covariance update. No
    branching on sensor presence is needed anywhere.
    """
    N = gyro_p.shape[0]
    R_pc = np.empty((N, 3, 3))
    dt2 = dt * dt

    P = P0.copy()
    q_wp = q_wp0.copy()
    q_wc = q_wc0.copy()

    Rp = np.empty((3, 3))
    Rc = np.empty((3, 3))
    Jp = np.empty((3, 3))
    Jc = np.empty((3, 3))
    b11 = np.empty((3, 3))
    b12 = np.empty((3, 3))
    b22 = np.empty((3, 3))
    dq_p = np.empty(4)
    dq_c = np.empty(4)
    qtmp = np.empty(4)
    H = np.empty((MEAS_DIM, 6))
    S = np.zeros((MEAS_DIM, MEAS_DIM))
    L = np.zeros((MEAS_DIM, MEAS_DIM))
    e = np.empty(MEAS_DIM)
    PHt = np.empty((6, MEAS_DIM))
    K = np.empty((6, MEAS_DIM))
    Pt = np.empty((6, 6))
    KH = np.empty((6, 6))
    n = np.empty(6)
    aa = np.empty(3)
    bb = np.empty(3)

    _quat_to_mat(q_wp, Rp)
    _quat_to_mat(q_wc, Rc)
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += Rp[k, i] * Rc[k, j]
            R_pc[0, i, j] = s

    for t in range(1, N):
        # ------------------------------ time update ------------------------------
        _rotvec_to_quat(dt * gyro_p[t - 1, 0], dt * gyro_p[t - 1, 1],
                        dt * gyro_p[t - 1, 2], dq_p)
        _rotvec_to_quat(dt * gyro_c[t - 1, 0], dt * gyro_c[t - 1, 1],
                        dt * gyro_c[t - 1, 2], dq_c)
        _quat_to_mat(dq_p, Jp)
        _quat_to_mat(dq_c, Jc)

        _congruence(Jp, Jp, P, 0, 0, b11)
        _congruence(Jp, Jc, P, 0, 3, b12)
        _congruence(Jc, Jc, P, 3, 3, b22)
        for i in range(3):
            for j in range(3):
                P[i, j] = b11[i, j]
                P[i, 3 + j] = b12[i, j]
                P[3 + j, i] = b12[i, j]
                P[3 + i, 3 + j] = b22[i, j]
        for i in range(6):
            for j in range(6):
                P[i, j] += dt2 * Q[i, j]

        _quat_mul(q_wp, dq_p, qtmp)
        for i in range(4):
            q_wp[i] = qtmp[i]
        _quat_mul(q_wc, dq_c, qtmp)
        for i in range(4):
            q_wc[i] = qtmp[i]
        _quat_to_mat(q_wp, Rp)
        _quat_to_mat(q_wc, Rc)

        # --------------------------- measurement update ---------------------------
        for i in range(MEAS_DIM):
            for j in range(MEAS_DIM):
                S[i, j] = 0.0

        for si in range(NUM_VECTOR_SENSORS):
            aa[0] = vp[t, si, 0]
            aa[1] = vp[t, si, 1]
            aa[2] = vp[t, si, 2]
            bb[0] = vc[t, si, 0]
            bb[1] = vc[t, si, 1]
            bb[2] = vc[t, si, 2]
            if normalize:
                na = np.sqrt(aa[0] ** 2 + aa[1] ** 2 + aa[2] ** 2)
                if na > 0.0:
                    aa[0] /= na
                    aa[1] /= na
                    aa[2] /= na
                nb = np.sqrt(bb[0] ** 2 + bb[1] ** 2 + bb[2] ** 2)
                if nb > 0.0:
                    bb[0] /= nb
                    bb[1] /= nb
                    bb[2] /= nb

            r = 3 * si
            for i in range(3):
                pw = Rp[i, 0] * aa[0] + Rp[i, 1] * aa[1] + Rp[i, 2] * aa[2]
                cw = Rc[i, 0] * bb[0] + Rc[i, 1] * bb[1] + Rc[i, 2] * bb[2]
                e[r + i] = pw - cw
                # row i of  R_wp @ skew(a).T  is  a x R_wp[i, :]
                H[r + i, 0] = aa[1] * Rp[i, 2] - aa[2] * Rp[i, 1]
                H[r + i, 1] = aa[2] * Rp[i, 0] - aa[0] * Rp[i, 2]
                H[r + i, 2] = aa[0] * Rp[i, 1] - aa[1] * Rp[i, 0]
                # row i of  -R_wc @ skew(b).T  is  -(b x R_wc[i, :])
                H[r + i, 3] = -(bb[1] * Rc[i, 2] - bb[2] * Rc[i, 1])
                H[r + i, 4] = -(bb[2] * Rc[i, 0] - bb[0] * Rc[i, 2])
                H[r + i, 5] = -(bb[0] * Rc[i, 1] - bb[1] * Rc[i, 0])

            # M R M^T for this sensor: R_wp diag(var_p) R_wp^T + R_wc diag(var_c) R_wc^T.
            # Cross-sensor blocks are zero, so only the diagonal block is written.
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for k in range(3):
                        s += Rp[i, k] * var_p[si, k] * Rp[j, k]
                        s += Rc[i, k] * var_c[si, k] * Rc[j, k]
                    S[r + i, r + j] = s

        for i in range(6):
            for j in range(MEAS_DIM):
                s = 0.0
                for k in range(6):
                    s += P[i, k] * H[j, k]
                PHt[i, j] = s                                  # P @ H.T
        for i in range(MEAS_DIM):
            for j in range(MEAS_DIM):
                s = 0.0
                for k in range(6):
                    s += H[i, k] * PHt[k, j]
                S[i, j] += s                                   # + H P H.T

        # K = PHt @ inv(S), one Cholesky solve per row of PHt
        if not _solve_spd(S, PHt, L, K):
            Sinv = np.linalg.inv(S)
            for i in range(6):
                for j in range(MEAS_DIM):
                    s = 0.0
                    for k in range(MEAS_DIM):
                        s += PHt[i, k] * Sinv[k, j]
                    K[i, j] = s

        for i in range(6):
            for j in range(6):
                s = 0.0
                for k in range(MEAS_DIM):
                    s += K[i, k] * H[k, j]
                KH[i, j] = (1.0 if i == j else 0.0) - s
        for i in range(6):
            for j in range(6):
                s = 0.0
                for k in range(6):
                    s += KH[i, k] * P[k, j]
                Pt[i, j] = s                                   # P_tilde
        for i in range(6):
            s = 0.0
            for k in range(MEAS_DIM):
                s += K[i, k] * e[k]
            n[i] = -s

        _rotvec_to_quat(n[0], n[1], n[2], dq_p)
        _rotvec_to_quat(n[3], n[4], n[5], dq_c)
        _quat_mul(q_wp, dq_p, qtmp)
        for i in range(4):
            q_wp[i] = qtmp[i]
        _quat_mul(q_wc, dq_c, qtmp)
        for i in range(4):
            q_wc[i] = qtmp[i]
        _quat_to_mat(dq_p, Jp)
        _quat_to_mat(dq_c, Jc)

        _congruence(Jp, Jp, Pt, 0, 0, b11)
        _congruence(Jp, Jc, Pt, 0, 3, b12)
        _congruence(Jc, Jc, Pt, 3, 3, b22)
        for i in range(3):
            for j in range(3):
                P[i, j] = b11[i, j]
                P[i, 3 + j] = b12[i, j]
                P[3 + j, i] = b12[i, j]
                P[3 + i, 3 + j] = b22[i, j]

        _quat_to_mat(q_wp, Rp)
        _quat_to_mat(q_wc, Rc)
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += Rp[k, i] * Rc[k, j]
                R_pc[t, i, j] = s

    return R_pc


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def run_relative_filter(gyro_p: np.ndarray, gyro_c: np.ndarray,
                        sensors_p: np.ndarray, sensors_c: np.ndarray,
                        dt: float,
                        gyro_std_parent: np.ndarray, gyro_std_child: np.ndarray,
                        vector_sensor_stds_parent: List[np.ndarray],
                        vector_sensor_stds_child: List[np.ndarray],
                        R_wp0: np.ndarray, R_wc0: np.ndarray,
                        normalize_measurements: bool = False,
                        init_orientation_std: float = np.deg2rad(0.1)) -> np.ndarray:
    """Run the whole trial and return R_pc, shape (N, 3, 3).

    Mirrors RelativeFilter(...) + set_qs(R_wp0, R_wc0) followed by one update() per
    sample, with the same index convention the pipeline uses: R_pc[0] is the seeded
    state, and step t consumes gyro[t-1] with sensors[t].

    sensors_p / sensors_c are (N, 2, 3), ordered [accelerometer, magnetometer] to
    match vector_sensor_stds_*.

    Raises NotImplementedError for any configuration this kernel does not cover, so a
    caller cannot silently get the wrong filter. Use RelativeFilter directly for joint
    constraints or a different sensor count.
    """
    if not NUMBA_AVAILABLE:                                      # pragma: no cover
        raise ImportError("relative_filter_fast requires numba; "
                          "callers should check NUMBA_AVAILABLE and fall back to "
                          "src.RelativeFilterPlus.RelativeFilter.")

    if len(vector_sensor_stds_parent) != NUM_VECTOR_SENSORS or \
            len(vector_sensor_stds_child) != NUM_VECTOR_SENSORS:
        raise NotImplementedError(
            f"This kernel covers exactly {NUM_VECTOR_SENSORS} vector sensors; got "
            f"{len(vector_sensor_stds_parent)}/{len(vector_sensor_stds_child)}. "
            f"Use RelativeFilter for other configurations.")

    gyro_p = np.ascontiguousarray(gyro_p, dtype=np.float64)
    gyro_c = np.ascontiguousarray(gyro_c, dtype=np.float64)
    vp = np.ascontiguousarray(sensors_p, dtype=np.float64)
    vc = np.ascontiguousarray(sensors_c, dtype=np.float64)
    if vp.shape != (gyro_p.shape[0], NUM_VECTOR_SENSORS, 3) or vp.shape != vc.shape:
        raise ValueError(f"sensors_p/sensors_c must be (N, {NUM_VECTOR_SENSORS}, 3) "
                         f"matching gyro; got {vp.shape} and {vc.shape}.")

    Q = np.diag(np.concatenate([np.asarray(gyro_std_parent, dtype=np.float64),
                                np.asarray(gyro_std_child, dtype=np.float64)]) ** 2)
    var_p = np.ascontiguousarray(
        [np.asarray(s, dtype=np.float64) ** 2 for s in vector_sensor_stds_parent])
    var_c = np.ascontiguousarray(
        [np.asarray(s, dtype=np.float64) ** 2 for s in vector_sensor_stds_child])

    from scipy.spatial.transform import Rotation
    q_wp0 = Rotation.from_matrix(R_wp0).as_quat().astype(np.float64)
    q_wc0 = Rotation.from_matrix(R_wc0).as_quat().astype(np.float64)
    P0 = np.eye(6) * init_orientation_std ** 2

    return _kernel(gyro_p, gyro_c, vp, vc, float(dt), Q, var_p, var_c,
                   q_wp0, q_wc0, P0, bool(normalize_measurements))
