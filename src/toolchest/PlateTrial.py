import os
import numpy as np
from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace
from typing import Tuple, List, Dict, Union
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _skew(vectors: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 3, 3) cross-product matrices."""
    zero = np.zeros(len(vectors))
    return np.stack([
        np.stack([zero, -vectors[:, 2], vectors[:, 1]], axis=1),
        np.stack([vectors[:, 2], zero, -vectors[:, 0]], axis=1),
        np.stack([-vectors[:, 1], vectors[:, 0], zero], axis=1)], axis=1)


# An order-4 Butterworth run through filtfilt needs more than 27 samples of padding, and a
# run barely longer than that carries no resolvable low-frequency content anyway.
_MIN_FILTER_RUN = 60

# Frames discarded at each end of a valid run, in periods of the analysis cutoff. Covers both
# the filter's own edge transient and the far larger spike that double-differentiating a
# position step produces at a gap boundary.
_EDGE_TRANSIENT_PERIODS = 4.0


def _contiguous_runs(mask: np.ndarray):
    """(start, stop) for each maximal run of True, so filters never cross a gap."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def _lowpass(values: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    """Zero-phase Butterworth along axis 0, shape preserved."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff_hz, btype='low', fs=sample_rate, output='sos')
    flat = sosfiltfilt(sos, values.reshape(len(values), -1), axis=0)
    return flat.reshape(values.shape)


class PlateTrial:
    """
    Contains a synchronized time-series of motion data for a single rigid body.
    
    A "Plate" represents a rigid body (e.g., a segment of a limb) that is
    tracked by both an IMU sensor and an external motion capture system (like
    Vicon or OptiTrack). This class stores the synchronized data from both
    sources.

    Attributes:
        name (str): A descriptive name for the plate (e.g., 'femur_r_imu').
        imu_trace (IMUTrace): An object containing the time-series data from the
            IMU sensor (accelerometer, gyroscope, magnetometer) in the
            sensor's local coordinate frame.
        world_trace (WorldTrace): An object containing the time-series data from
            the motion capture system (position, orientation) in the
            global coordinate frame.
    """

    name: str
    imu_trace: IMUTrace
    world_trace: WorldTrace

    def __init__(self, name: str, imu_trace: IMUTrace, world_trace: WorldTrace):
        """
        Initializes a new PlateTrial object.

        Args:
            name (str): The name for this plate trial.
            imu_trace (IMUTrace): The IMU data trace.
            world_trace (WorldTrace): The World (mocap) data trace.

        Raises:
            AssertionError: If the IMU and World traces have different lengths
                or their timestamps do not match within a small tolerance (1e-8).
        """
        # Ensure data consistency
        assert len(imu_trace) == len(world_trace), \
            "IMU and World traces must have the same length"
        
        # Check for timestamp synchronization
        if max(np.abs(imu_trace.timestamps - world_trace.timestamps)) > 1e-8:
            if os.getenv("DISABLE_TQDM", "False") != "True":
                print(f"Warning: IMU and World traces must have the same timestamps. "
                      f"Max difference: {max(np.abs(imu_trace.timestamps - world_trace.timestamps))}")
        
        assert max(np.abs(imu_trace.timestamps - world_trace.timestamps)) < 1e-8, \
            "Timestamps must match"
        
        # Type checking
        assert isinstance(imu_trace, IMUTrace)
        assert isinstance(world_trace, WorldTrace)

        self.name = name
        # Where this plate's world trace was moved to, in the sensor frame, metres. Zero means
        # the pose still describes the marker cluster rather than the IMU. Set by
        # assembly.shift_world_origin; not carried through the parquet, which is why the
        # build records it in the manifest instead.
        self.sensor_offset = np.zeros(3)
        self.imu_trace = imu_trace
        self.world_trace = world_trace

    def __len__(self) -> int:
        """Returns the number of samples (timesteps) in the trial."""
        return len(self.imu_trace)

    @property
    def sample_rate(self) -> float:
        """Samples per second. NOT the same for every trial, and not always 100.

        The loader keeps each trial at the slowest rate any of its streams was recorded at,
        rather than resampling everything onto one global grid. Al Borno's mocap is 100 Hz
        throughout but Subject06 and Subject10 carry 40 Hz IMUs, so those three trials load
        at 40 Hz; IMoVE is 40 Hz for the 17-sensor sessions and 100 Hz for the 7-sensor
        long-walk ones. Upsampling to hide that would be inventing data -- see
        toolchest.resampling for what it cost when the loader used to do exactly that.

        So anything with a time constant -- a filter cutoff, a smoothing window, a process
        noise -- has to be expressed per SECOND and converted through this, not baked in as
        a sample count. `finite_difference_utils.window_samples` is the worked example.
        """
        return float(self.imu_trace.get_sample_frequency())

    @property
    def valid(self) -> np.ndarray:
        """Per-frame boolean: is this sample usable as GROUND TRUTH?

        False where the marker reconstruction interpolated a pose or left a corrupt one
        in place (see WorldTrace.repair_reconstruction_glitches). The arrays themselves are
        never gapped — a filter still runs straight through these frames, because the
        uniform time grid is what makes its integration and every finite difference in this
        repo well defined. The mask is what keeps a manufactured pose out of the error
        statistics computed against it.

        Delegates to the world trace because ground-truth validity is a mocap property. If
        IMU-side validity is ever needed — the IMoVE BioStamp data has a saturated
        accelerometer channel and dropped-sample logs — this is where the two would be
        AND-ed, and every consumer of `plate.valid` would pick it up unchanged.
        """
        return self.world_trace.valid

    def __getitem__(self, key: slice) -> 'PlateTrial':
        """
        Enables slicing of the PlateTrial object.

        This allows you to take a subset of the time-series data, e.g., `trial[100:500]`.

        Args:
            key (slice): The slice object specifying the start, stop, and step.

        Returns:
            PlateTrial: A new PlateTrial object containing only the sliced data.
        """
        if not isinstance(key, slice):
            raise TypeError("PlateTrial slicing only supports slice objects.")
        return PlateTrial(self.name, self.imu_trace.copy()[key], self.world_trace.copy()[key])
    
    def copy(self) -> 'PlateTrial':
        """
        Returns a deep copy of the PlateTrial object.

        This creates new IMUTrace and WorldTrace objects with copied data,
        preventing modifications to the copy from affecting the original.

        Returns:
            PlateTrial: A new, independent copy of the object.
        """
        return PlateTrial(
            self.name,
            self.imu_trace.copy(),
            self.world_trace.copy()
        )

    def fit_sensor_offset(self, lowpass_hz: float = 8.0,
                          excitation_percentile: float = 50.0,
                          omega_from: str = 'gyro') -> np.ndarray:
        """Where the IMU sits relative to the mocap origin, in metres, in the SENSOR frame.

        Two rigidly connected points on one body differ only by the lever-arm term, so

            R f_imu - f_mocap  =  A p,     A = [alpha]x + omega omega^T - |omega|^2 I

        is linear in p with three unknowns and thousands of equations.

        GRAVITY IS SOLVED FOR, NOT SUPPLIED. Taking the mocap's specific force with no gravity
        term at all leaves R^T a_origin, so the residual against the real accelerometer is

            b = f_imu - R^T a_origin  =  R^T gamma  +  A p

        with gamma the gravity contribution to the reading. That is linear in gamma AND p
        together, so stacking [R^T | A] recovers both from six unknowns against thousands of
        equations. The alternative -- estimating gamma as the mean rotated accelerometer
        reading and subtracting it -- makes the answer depend on that estimate, and any error
        in it leaves a residue proportional to R^T, which is correlated with orientation and
        therefore lands squarely in p. Solving jointly removes the dependence entirely, and
        the recovered |gamma| is a free check: it has to come back at about 9.81.

        LOW-PASSED BY DEFAULT, and that is not cosmetic. A is built from twice-differentiated
        marker positions, so its noise grows as f^2 while the accelerometer's does not; the
        two sides of the equation disagree most exactly where the motion is fastest. Filtering
        is EXACT here rather than approximate, because p is a constant: for any linear filter
        L, L[A p] = L[A] p = L[b], so it is the same constraint restricted to a band where the
        rigid-body model holds. Measured on IMoVE, 8 Hz roughly halves both the trial-to-trial
        spread and the repeat scatter; below 5 Hz A shrinks and the system goes ill-conditioned.

        Only the most excited half of the frames are used, because A p is what carries the
        signal and frames where A is small contribute noise in proportion.

        The result is in the sensor frame whether or not the plate was aligned at load time:
        the sensor-to-segment rotation is re-solved from the gyros here, and is the identity
        when assembly has already applied it.
        """
        from .gyro_utils import calculate_best_fit_rotation

        valid = np.asarray(self.valid)
        if valid.sum() < 100:
            raise ValueError(f"{self.name}: only {valid.sum()} valid frames to fit against.")

        # No gravity term: the residual below then carries it, and it is solved for.
        synthetic = self.world_trace.calculate_imu_trace()

        rotation = calculate_best_fit_rotation(synthetic.gyro[valid],
                                               self.imu_trace.gyro[valid])
        residual = (self.imu_trace.acc @ rotation.T) - synthetic.acc

        # WHERE OMEGA COMES FROM decides how noisy A is, and A is the design matrix, so its
        # noise biases p toward zero rather than merely scattering it.
        #
        #   'gyro'   uses the gyroscope's own reading, which IS omega, measured. One
        #            differentiation instead of two, and no marker-reconstruction noise at all.
        #   'mocap'  differentiates the reconstructed pose once for omega and twice for alpha.
        #            Kept for comparison; this was the default until it was measured against
        #            the alternative and lost.
        #
        # 'gyro' is the default on two measurements over IMoVE's 100 Hz sessions. Sensitivity
        # of the answer to the analysis cutoff, which a rigid lever arm should not have at all,
        # falls from 77% to 26% -- and on THIGH_R_M from 206% to 26%. And left/right agreement,
        # which is a genuine test because the two sides are independent hardware fitted
        # independently, improves from 2.3 to 1.1 mm on the shanks and from 2.9 to 0.3 mm on
        # the thighs.
        #
        # The frames are already equivalent: after assembly.align_world_to_imu the world trace's
        # rotations map sensor -> world, so calculate_imu_trace's gyro and the measured gyro are
        # both in the sensor frame and `rotation` below is the identity. What differs is not the
        # frame but the SIGNAL -- the two disagree by 10-25% RMS, and that discrepancy is
        # marker-reconstruction noise, sync error and soft tissue.
        #
        # The honest cost: `residual` still takes the origin's acceleration from mocap, which
        # the gyro cannot supply, so A and b now come from different sources. The old path was
        # internally consistent but noisy; this one is cross-sourced but far more accurate in
        # A, which is the design matrix and therefore the term whose noise biases p.
        #
        # Mocap is still needed for `residual`, which wants the origin's linear acceleration --
        # something the gyro cannot supply. So this moves only the design matrix off the mocap,
        # not the whole fit.
        if omega_from == 'gyro':
            source = IMUTrace(self.imu_trace.timestamps,
                              self.imu_trace.gyro @ rotation.T,
                              self.imu_trace.acc, self.imu_trace.mag)
        elif omega_from == 'mocap':
            source = synthetic
        else:
            raise ValueError(f"omega_from must be 'mocap' or 'gyro', got {omega_from!r}.")

        omega = source.gyro
        alpha = source._finite_difference_gyros('polyfit')
        lever = (_skew(alpha) + np.einsum('ni,nj->nij', omega, omega)
                 - (omega ** 2).sum(axis=1)[:, None, None] * np.eye(3))
        # Body-frame gravity direction per sample, the other half of the design matrix.
        # COPIED, not a view: transpose returns a view onto the world trace's own rotations,
        # and the filtering below writes in place. Without the copy this method silently
        # corrupts the plate it was called on -- and because a segment's three sensors share
        # one WorldTrace, fitting the H sensor would have poisoned M and L behind it.
        orientation = np.asarray(self.world_trace.rotations).transpose(0, 2, 1).copy()

        if lowpass_hz is not None and lowpass_hz < 0.5 * self.sample_rate:
            # Legitimate on every block because gamma and p are both CONSTANT, so a linear
            # filter passes straight through them: L[R^T gamma + A p] = L[R^T] gamma + L[A] p.
            #
            # Filtered RUN BY RUN, never across a gap. Invalid stretches hold a constant pose,
            # so the boundary into one is a step in acceleration, and filtering over it drags
            # the step back into real data. On IMoVE's long walks -- one inertial record
            # spanning three mocap takes, hence six such boundaries -- that alone moved the
            # recovered offset from 13 mm to 200 mm.
            # Each run's EDGES are discarded as well. Two transients meet there: the filter's
            # own, and a much larger one from upstream -- calculate_imu_trace differentiates
            # the whole position array twice before any of this, so the step between a held
            # gap and the next real pose spikes the frames either side of it. On IMoVE's long
            # walks that boundary appears six times per trial.
            edge = int(np.ceil(_EDGE_TRANSIENT_PERIODS * self.sample_rate / lowpass_hz))
            filterable = np.zeros(len(valid), dtype=bool)
            for start, stop in _contiguous_runs(valid):
                if stop - start < _MIN_FILTER_RUN:
                    continue
                for block in (lever, orientation, residual):
                    block[start:stop] = _lowpass(block[start:stop], lowpass_hz,
                                                 self.sample_rate)
                trim = min(edge, (stop - start) // 4)
                filterable[start + trim:stop - trim] = True
            valid = valid & filterable
            if not valid.any():
                raise ValueError(f"{self.name}: no run of valid frames is long enough to "
                                 f"filter at {lowpass_hz} Hz.")

        strength = np.linalg.norm(lever, axis=(1, 2))
        used = valid & (strength >= np.percentile(strength[valid], excitation_percentile))

        design = np.concatenate([orientation[used], lever[used]], axis=2)
        solution, *_ = np.linalg.lstsq(design.reshape(-1, 6),
                                       residual[used].reshape(-1), rcond=None)
        self._fitted_gravity = solution[:3]
        # Back into the sensor's own frame; a no-op when the plate was aligned at load.
        return rotation.T @ solution[3:]

    def project_imu_trace(self, local_offset: np.ndarray) -> IMUTrace:
        r"""
        Estimates the IMUTrace values at a different location on the same rigid body.
        Currently only projects accelerometer data, but could be extended to gyro/mag.

        This uses the rigid body equations of motion to project the accelerometer
        data to a new point, defined by the `local_offset` vector in the
        IMU's coordinate frame.
        
        $a_p = a_c + \alpha \times r + \omega \times (\omega \times r)$
        
        where:
        - $a_c$ is the measured acceleration at the IMU center.
        - $\omega$ is the measured angular velocity.
        - $\alpha$ is the angular acceleration (derived from gyro).
        - $r$ is the `local_offset` vector (from IMU center to new point P).
        - $a_p$ is the projected acceleration at point P.

        Args:
            local_offset (np.ndarray): A (3,) array representing the 3D
                vector from the current IMU location to the new,
                projected location, expressed in the IMU's local frame.

        Returns:
            IMUTrace: A new IMUTrace object with the projected sensor data.
        """
        # This function relies on the IMUTrace class to handle the physics.
        return self.imu_trace.project_acc(local_offset)
    
    def get_imu_trace_in_global_frame(self) -> IMUTrace:
        """
        Rotates the IMU sensor data into the global coordinate frame.

        This uses the (ground truth) `world_trace.rotations` to transform
        the accelerometer, gyroscope, and magnetometer vectors from the
        sensor's local frame to the global frame at each timestep.

        Returns:
            IMUTrace: A new IMUTrace object where all data vectors are
                expressed in the global coordinate frame.
        """
        # R is the rotation from local-to-global (from world_trace)
        # v_global = R @ v_local
        world_rots = self.world_trace.rotations
        
        rotated_acc = np.einsum('nij,nj->ni', world_rots, self.imu_trace.acc)
        rotated_gyro = np.einsum('nij,nj->ni', world_rots, self.imu_trace.gyro)
        rotated_mag = np.einsum('nij,nj->ni', world_rots, self.imu_trace.mag)
        
        return IMUTrace(
            timestamps=self.imu_trace.timestamps,
            acc=rotated_acc,
            gyro=rotated_gyro,
            mag=rotated_mag
        )
    
    def find_biaxial_joint_axes(
        self: 'PlateTrial',
        other: 'PlateTrial',
        initial_axis_parent: np.ndarray = None,
        initial_axis_child: np.ndarray = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        subsample_rate: int = 1,
        verbose: bool = False # ⬅️ ADDED VERBOSE FLAG
    ) -> Dict[str, Union[np.ndarray, bool]]:
        r"""
        Estimates the two joint axes for a biaxial (2-DoF) joint, where each
        axis is constant in its respective local segment frame.

        This algorithm correctly models a biological joint where the axes are
        fixed to the body segments. The core principle is that the relative
        angular velocity, when expressed in the world frame, must lie in the
        plane defined by the two joint axes, which are also expressed in the
        world frame.

        The unknowns being solved for are the constant representations of the
        axes in their local frames:
        - j1_parent: The first axis, constant in the 'self' (parent) frame.
        - j2_child: The second axis, constant in the 'other' (child) frame.

        The cost function at each timestep 't' is:
        e(t) = (w_child_w(t) - w_parent_w(t)) . (j1_w(t) x j2_w(t))
        where:
        - j1_w(t) = R_parent(t) @ j1_parent
        - j2_w(t) = R_child(t) @ j2_child

        Args:
            other (PlateTrial): The PlateTrial for the 'child' segment. 'self'
                is treated as the 'parent' segment.
            initial_axis_parent (np.ndarray, optional): A (3,) initial guess for
                the first joint axis in the PARENT's local frame.
            initial_axis_child (np.ndarray, optional): A (3,) initial guess for
                the second joint axis in the CHILD's local frame.
            max_iterations (int, optional): Max iterations for the optimization.
            tolerance (float, optional): Convergence tolerance.
            subsample_rate (int, optional): Rate to subsample data.
            verbose (bool, optional): If True, prints status updates during the
                                    optimization loop. ⬅️ ADDED DOC FOR VERBOSE

        Returns:
            Dict[str, Union[np.ndarray, bool]]: A dictionary containing:
                - 'axis_parent_local': The optimized (3,) unit axis vector in the parent frame.
                - 'axis_child_local': The optimized (3,) unit axis vector in the child frame.
                - 'converged': A boolean flag indicating if convergence was reached.
        """
        # 1. --- Input Validation and Data Preparation ---
        np.testing.assert_array_almost_equal(
            self.imu_trace.timestamps, other.imu_trace.timestamps, decimal=5,
            err_msg="PlateTrial traces must be time-synchronized."
        )
        indices = np.arange(0, len(self.imu_trace.timestamps), subsample_rate) # Use imu_trace length here

        # 2. --- Transform Gyro Data and Get Rotations ---
        if verbose:
            print("--- Joint Axis Estimation (Biaxial) ---")
            print(f"Subsampling data to {len(indices)} points from {len(self.imu_trace.timestamps)} total.")
            
        p_imu_trace = self.get_imu_trace_in_global_frame()
        g_p_world = p_imu_trace.gyro[indices]
        R_wp = self.world_trace.rotations[indices]

        c_imu_trace = other.get_imu_trace_in_global_frame()
        g_c_world = c_imu_trace.gyro[indices]
        R_wc = other.world_trace.rotations[indices]

        w_rel_world = g_c_world - g_p_world

        # 3. --- Helper Functions for Parametrization ---
        def spherical_to_cartesian(phi, theta):
            return np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)])

        def cartesian_derivatives(phi, theta):
            return (np.array([-np.sin(phi)*np.cos(theta), -np.sin(phi)*np.sin(theta), np.cos(phi)]),
                    np.array([-np.cos(phi)*np.sin(theta), np.cos(phi)*np.cos(theta), 0]))

        def cartesian_to_spherical(axis: np.ndarray):
            phi = np.arcsin(np.clip(axis[2], -1.0, 1.0))
            theta = np.arctan2(axis[1], axis[0])
            return phi, theta

        # 4. --- Initialization of State Vector ---
        # State vector x = [phi1, theta1, phi2, theta2] for j1_parent and j2_child
        if initial_axis_parent is None:
            phi1, theta1 = np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-np.pi, np.pi)
        else:
            phi1, theta1 = cartesian_to_spherical(initial_axis_parent/np.linalg.norm(initial_axis_parent))

        if initial_axis_child is None:
            phi2, theta2 = np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-np.pi, np.pi)
        else:
            phi2, theta2 = cartesian_to_spherical(initial_axis_child/np.linalg.norm(initial_axis_child))

        x = np.array([phi1, theta1, phi2, theta2])
        
        if verbose:
            print(f"Initial State Vector x: {x}")

        # 5. --- Gauss-Newton Optimization Loop ---
        for i in range(max_iterations): # Used 'i' for iteration number
            phi1, theta1, phi2, theta2 = x
            j1_p = spherical_to_cartesian(phi1, theta1)
            j2_c = spherical_to_cartesian(phi2, theta2)
            dj1p_dphi1, dj1p_dtheta1 = cartesian_derivatives(phi1, theta1)
            dj2c_dphi2, dj2c_dtheta2 = cartesian_derivatives(phi2, theta2)

            # Transform local axes to world frame FOR EACH TIMESTEP
            j1_w_t = np.einsum('nij,j->ni', R_wp, j1_p)
            j2_w_t = np.einsum('nij,j->ni', R_wc, j2_c)

            jn_w_t = np.cross(j1_w_t, j2_w_t)
            norm_jn_t = np.linalg.norm(jn_w_t, axis=1, keepdims=True)

            # Avoid division by zero
            if np.any(norm_jn_t < 1e-9):
                if verbose:
                    print(f"Iteration {i+1}: 🛑 Warning: Joint axes became parallel (angle ~0 or ~180). Stopping.")
                break
            
            jn_w_norm_t = jn_w_t / norm_jn_t
            e = np.einsum('ni,ni->n', w_rel_world, jn_w_norm_t) # Residual (Error vector)

            # Calculate Residual Norm for monitoring
            residual_norm = np.linalg.norm(e)
            if verbose:
                print(f"Iteration {i+1}/{max_iterations}: Residual Norm (Cost): {residual_norm:.6e}", end="")


            # Jacobian 'J' (N, 4) using the chain rule (Calculations remain the same)
            dj1w_dphi1 = np.einsum('nij,j->ni', R_wp, dj1p_dphi1)
            dj1w_dtheta1 = np.einsum('nij,j->ni', R_wp, dj1p_dtheta1)
            dj2w_dphi2 = np.einsum('nij,j->ni', R_wc, dj2c_dphi2)
            dj2w_dtheta2 = np.einsum('nij,j->ni', R_wc, dj2c_dtheta2)

            # Derivatives of the plane normal (jn_w)
            djn_dphi1 = np.cross(dj1w_dphi1, j2_w_t)
            djn_dtheta1 = np.cross(dj1w_dtheta1, j2_w_t)
            djn_dphi2 = np.cross(j1_w_t, dj2w_dphi2)
            djn_dtheta2 = np.cross(j1_w_t, dj2w_dtheta2)
            
            # Final Jacobian columns
            J0 = np.einsum('ni,ni->n', w_rel_world, djn_dphi1 / norm_jn_t)
            J1 = np.einsum('ni,ni->n', w_rel_world, djn_dtheta1 / norm_jn_t)
            J2 = np.einsum('ni,ni->n', w_rel_world, djn_dphi2 / norm_jn_t)
            J3 = np.einsum('ni,ni->n', w_rel_world, djn_dtheta2 / norm_jn_t)

            J = np.column_stack((J0, J1, J2, J3))
            
            try:
                # Solve for update vector
                delta_x = -np.linalg.pinv(J) @ e
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Iteration {i+1}: 🛑 Warning: Singular matrix in pseudoinverse. Stopping.")
                break

            x_prev = x.copy()
            x += delta_x
            update_norm = np.linalg.norm(delta_x)

            if verbose:
                print(f" | Update Norm: {update_norm:.6e}")
                if i+1 == max_iterations:
                    print("--- Optimization finished: Max iterations reached ---")

            # Check for convergence
            if update_norm < tolerance:
                j1_res = spherical_to_cartesian(x[0], x[1])
                j2_res = spherical_to_cartesian(x[2], x[3])
                if verbose:
                    print(f"\n✅ CONVERGENCE REACHED in {i+1} iterations.")
                    print(f"Final Axis Parent (Local): {j1_res}")
                    print(f"Final Axis Child (Local): {j2_res}")
                return {'axis_parent_local': j1_res, 'axis_child_local': j2_res, 'converged': True}

        # 6. --- Return final (non-converged) estimate ---
        j1_res = spherical_to_cartesian(x[0], x[1])
        j2_res = spherical_to_cartesian(x[2], x[3])
        
        if verbose and update_norm >= tolerance:
            print("\n❌ CONVERGENCE FAILED: Tolerance not met.")
            print(f"Final Axis Parent (Local): {j1_res}")
            print(f"Final Axis Child (Local): {j2_res}")

        return {'axis_parent_local': j1_res, 'axis_child_local': j2_res, 'converged': False}
        
