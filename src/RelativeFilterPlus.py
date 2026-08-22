from typing import Tuple, List, Optional, Dict
import numpy as np
from scipy.spatial.transform import Rotation

from src import joint_constraints

JOINT_TYPES = ('1dof', '2dof', 'coupling', 'spring')

# Below this sine of the angle between the reference sensor and the heading-only sensor, the
# projection axis is dropped rather than normalized (see _heading_projection_axis).
#
# The sine IS the retained sensitivity: the residual's response to a heading error of theta is
# |m_w| sin(angle) * theta, so at the gate the row is already down to a tenth of its best case,
# and the axis DIRECTION is what degrades first -- its angular uncertainty is roughly the
# reading's relative noise divided by the sine, so ~1% magnetometer noise puts it at ~6 deg here
# and rising as 1/sine below. Normalizing past that hands the filter a confidently-signed
# correction about an essentially random axis, which is worse than no row at all.
#
# Al Borno's field sits 133-152 deg from the joint-centre accelerometer, i.e. sine 0.47-0.73,
# so this is a guard against transients rather than a duty cycle on that dataset.
HEADING_AXIS_MIN_SINE = 0.1


class RelativeFilter:
    """
    An Extended Kalman Filter to estimate the relative orientation between two
    segments (parent and child).

    It uses:
    1. Gyroscope data for prediction.
    2. Any number of 1DOF vector sensors (e.g., accelerometers) for update.
    3. An optional, mutually exclusive joint constraint.

    The state is the 6D error-state rotation vector eta = [eta_p, eta_c].

    THE FOUR JOINT CONSTRAINTS, in the order of the model ladder experiments/joint_dof.py
    measures. Each adds rows to the measurement vector; the count is what decides how much of
    the relative orientation it can pin, and with the magnetometer off the interesting question
    is how much of the UNOBSERVABLE direction (relative heading, the rotation about gravity)
    each row spans:

      '1dof'      hinge.      3 rows, rank 2. One axis matched in both frames.
      'coupling'  the knee.   2 rows. The two non-sagittal channels pinned to Reuben's
                              published polynomials of flexion rather than to zero.
      '2dof'      universal.  1 row. Only the ANGLE between two axes is fixed.
      'spring'    neutral.    1 row, deliberately weak. One channel pulled toward zero.

    'coupling' and 'spring' are written in the model's own coordinate nu = log(R_0^T R_pc),
    with R_0 the neutral relative rotation and every axis given in the CHILD frame — the
    convention experiments/joint_dof.py fits in. See src/joint_constraints.py.

    The constraint std is NOT the model's measured error, and setting it there is the mistake
    that makes a constraint look useless. Model error is soft-tissue motion and secondary
    rotation, correlated over many samples, so feeding its per-sample scatter in as white
    measurement noise wildly overstates the uncertainty in the constraint's MEAN direction. On
    a knee the useful std runs about an order of magnitude tighter than the measured error;
    over-trusting is punished at the other end, where a near-hard constraint on a slightly
    wrong axis drags the whole estimate with it. Sweep it.
    """
    
    q_wp: Rotation
    q_wc: Rotation
    
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    num_vector_sensors: int
    normalize_measurements: bool = False
    heading_only_sensor: Optional[int] = None
    heading_only_pure: bool = False

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
                 dof2_angle_rad: Optional[float] = None,
                 dof2_std: Optional[float] = None,
                 # Soft-model joint args, both written in nu = log(R_0^T R_pc)
                 model_R_0: Optional[np.ndarray] = None,
                 coupling_axis_child: Optional[np.ndarray] = None,
                 coupling_channel_map: Optional[np.ndarray] = None,
                 coupling_flexion_sign: float = 1.0,
                 coupling_std: Optional[float] = None,
                 spring_axis_child: Optional[np.ndarray] = None,
                 spring_std: Optional[float] = None,
                 normalize_measurements: bool = False,
                 heading_only_sensor: Optional[int] = None,
                 heading_only_pure: bool = False,
                 init_orientation_std: float = np.deg2rad(0.1)):
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
            normalize_measurements: If True, every vector sensor reading is scaled to unit
                                    length before the measurement update, so only its
                                    direction is used. The vector_sensor_stds are NOT
                                    rescaled to match — see _normalize_vector_measurements
                                    for what that does to the acc/gyro trust ratio.
            heading_only_sensor: Index of a vector sensor restricted to correcting the ONE
                                 relative DOF that vector sensor 0 cannot observe, or None
                                 for the usual full-vector update. In this pipeline sensor 0
                                 is the accelerometer and sensor 1 the magnetometer, so
                                 heading_only_sensor=1 is "let the magnetometer set heading
                                 and nothing else".

                                 WHY THERE IS A DOF NEITHER SENSOR SHARES. A relative rotation
                                 delta moves sensor 0's residual R_wp a_p - R_wc a_c by
                                 -delta x a_w, which vanishes exactly when delta is parallel to
                                 a_w. So the accelerometer pair pins the two relative DOF
                                 perpendicular to the joint-centre acceleration and carries no
                                 information at all about rotation ABOUT it. That axis -- the
                                 measured acceleration, not gravity, and they differ by tens of
                                 degrees under load -- is what the magnetometer alone can supply.
                                 Fed in as a full 3-vector it also votes on the two DOF the
                                 accelerometer already owns, at a weight that is not small.
                                 A vector sensor's authority per DOF it can see goes as
                                 (|v|/sigma_v)^2, and the default stds put the two within 2% of
                                 each other nominally (9.81/0.09695 = 101 against 1/0.009695 =
                                 103); measured on Al Borno the magnetometer lands at 0.72-0.86
                                 of the accelerometer, the readings being smaller than nominal.

                                 THE VOTE IT GIVES UP IS THE BAD ONE, which is why this is worth
                                 a knob. Scored against mocap rotations on Al Borno walking, so
                                 that every residual is pure sensor disagreement, the two
                                 sensors' demands for spurious tilt are comparable at full
                                 bandwidth (acc 12-29 deg RMS, mag 5-23) but not at all
                                 comparable once block-averaged over 1 s: the accelerometer's
                                 collapses to 0.7-1.6 deg, being heel-strike transients and
                                 lever-arm residual that the filter averages away, while the
                                 magnetometer's stays at 2.9-11.5 deg because field
                                 non-uniformity is quasi-static. In the DOF they share, the mag
                                 holds a 3-7x larger SUSTAINED error at equal weight.

            heading_only_pure: Strip the reference direction's component out of the restricted
                               sensor's reading before the residual and Jacobian are formed, so
                               that the single retained row is attributed to rotation about the
                               REFERENCE direction rather than about the sensor's own direction
                               with the reference mixed in. Requires heading_only_sensor.

                               IT CHANGES NO MEASUREMENT, ONLY THE JACOBIAN. The retained scalar
                               is w . e for w = unit(a_w x v_w), and w is perpendicular to a_w, so
                               stripping the a_w component out of the reading changes neither w
                               nor w . e -- both are bit-identical. What moves is the row
                               dh/d(delta):

                                 off:  direction t = unit(m x w), gain |v_w|
                                       t = sin(theta) a_hat - cos(theta) p_hat, so the row is
                                       attributed partly to a tilt direction the accelerometer
                                       already owns, and heading must be recovered by asking the
                                       accelerometer for the rest -- which amplifies the
                                       accelerometer's INSTANTANEOUS error into heading by
                                       |cot(theta)|.
                                 on:   direction a_hat exactly, gain |v_w| sin(theta).

                               So it buys a pure attribution at the price of a row trusted
                               sin(theta)^2 less in information terms (theta = 133-152 deg on Al
                               Borno, so 0.22-0.53x). Which of those dominates is not decidable
                               from the geometry and has to be measured.

                               Equivalently: this is what preprocessing the field to its
                               tilt-compensated component WOULD do if both segments projected
                               against one shared direction instead of each against its own
                               accelerometer. Doing it here rather than on the data is what makes
                               the shared direction available at all, and it avoids the own-axis
                               version's defect of manufacturing a magnetometer residual out of a
                               disagreement between the two accelerometers.

            init_orientation_std: Per-axis std dev (rad) of the initial orientation state,
                                  used to build P. The default assumes the caller seeds the
                                  state with set_qs from a known orientation; leaving P at
                                  eye(6) (~57 deg/axis) instead makes the first measurement
                                  update apply nearly the whole residual as a correction,
                                  which shows up as a tens-of-seconds startup transient —
                                  far worse without the magnetometer, since relative heading
                                  is then only observable through motion.
            
            dof1_axis_parent: (3,) vector for 1DOF joint on parent (y^J in paper).
            dof1_axis_child: (3,) vector for 1DOF joint on child (y^K in paper).
            dof1_std: (3,) std dev of the 1DOF constraint (e_link in paper).
            
            dof2_axis_parent: (3,) axis vector for 2DOF constraint on parent (v^J in paper).
            dof2_axis_child: (3,) axis vector for 2DOF constraint on child (u^K in paper).
            dof2_angle_rad: The fixed angle (alpha) BETWEEN the 2DOF axes, so the constraint
                            enforced is (R_wp v^J).(R_wc u^K) = cos(alpha). A universal joint
                            with perpendicular axes is alpha = pi/2, i.e. dot product zero.

                            REQUIRED, deliberately: this used to default to pi/2 while the
                            code compared the dot product against SIN(alpha), which made the
                            default enforce a dot product of 1 — coincident axes, which is not
                            a universal joint under either reading of alpha. A caller that
                            relied on the default got a silently wrong constraint, so there is
                            no default to rely on any more.
            dof2_std: The std dev of the 2DOF scalar measurement (dot product).

            model_R_0: (3, 3) neutral relative rotation R_0, shared by the 'coupling' and
                       'spring' constraints. The model is R_pc = R_0 exp([nu]), so nu = 0 is
                       the neutral pose. `joint_dof` fits and persists this as 'R_0'.
            coupling_axis_child: (3,) flexion axis u_c in the CHILD frame ('axis_child' in
                       joint_dof's stored coupling fit).
            coupling_channel_map: (2, 2) map from Reuben's (adduction, rotation) channels into
                       the tangent basis of u_c. joint_dof searches this per subject because
                       dropping the published channels into an arbitrary basis unaligned
                       scored the same curve at 23.2 deg instead of 15.1.
            coupling_flexion_sign: +1 or -1, the sense of flexion in this plate's frame.
            coupling_std: std dev of EACH of the two coupling channels, in radians.

            spring_axis_child: (3,) the channel pulled toward neutral, in the child frame —
                       internal/external rotation at the hip and ankle.
            spring_std: std dev of the spring measurement, in radians. Deliberately large;
                       `joint_constraints.spring_std_for_gain` converts IMoveLab's feedback
                       gain kappa into a comparable value.
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
        if heading_only_sensor is not None:
            if not 0 < heading_only_sensor < self.num_vector_sensors:
                raise ValueError(
                    f"heading_only_sensor={heading_only_sensor} is not a vector sensor other "
                    f"than 0; there are {self.num_vector_sensors} sensors and sensor 0 is the "
                    f"reference whose unobservable axis the restriction is defined against. "
                    f"Restricting sensor 0 to its own null direction would delete it.")
        if heading_only_pure and heading_only_sensor is None:
            raise ValueError(
                "heading_only_pure has no meaning without heading_only_sensor: it changes which "
                "rotation direction the restricted sensor's single row is attributed to, and "
                "without the restriction there is no single row.")
        self.heading_only_sensor = heading_only_sensor
        self.heading_only_pure = heading_only_pure
        self.joint_type = joint_type
        self.joint_params = {}
        self.normalize_measurements = normalize_measurements

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
            if (dof2_axis_parent is None or dof2_axis_child is None or dof2_std is None
                    or dof2_angle_rad is None):
                raise ValueError("For '2dof' joint, must provide dof2_axis_parent, "
                                 "dof2_axis_child, dof2_angle_rad, and dof2_std.")
            if dof2_axis_parent.shape != (3,) or dof2_axis_child.shape != (3,):
                 raise ValueError("2DOF joint axes must have shape (3,)")

            self.joint_params = {
                'v_j': dof2_axis_parent / np.linalg.norm(dof2_axis_parent),
                'u_k': dof2_axis_child / np.linalg.norm(dof2_axis_child),
                'cos_alpha': np.cos(dof2_angle_rad)
            }
            # This constraint's noise is 1D (scalar)
            all_variances.append(np.array([dof2_std**2]))
            self.joint_noise_dim = 1

        elif self.joint_type == 'coupling':
            if (coupling_axis_child is None or model_R_0 is None or coupling_std is None
                    or coupling_channel_map is None):
                raise ValueError("For 'coupling' joint, must provide coupling_axis_child, "
                                 "model_R_0, coupling_channel_map, and coupling_std.")
            if coupling_axis_child.shape != (3,):
                raise ValueError("coupling_axis_child must have shape (3,)")
            if np.asarray(model_R_0).shape != (3, 3):
                raise ValueError("model_R_0 must have shape (3, 3)")
            if np.asarray(coupling_channel_map).shape != (2, 2):
                raise ValueError("coupling_channel_map must have shape (2, 2)")
            if coupling_flexion_sign not in (1.0, -1.0, 1, -1):
                raise ValueError(f"coupling_flexion_sign must be +1 or -1, "
                                 f"got {coupling_flexion_sign}")

            u_c = coupling_axis_child / np.linalg.norm(coupling_axis_child)
            self.joint_params = {
                'u_c': u_c,
                'R_0': np.asarray(model_R_0, dtype=float),
                'basis_P': joint_constraints.so3.tangent_basis(u_c),
                'channel_map': np.asarray(coupling_channel_map, dtype=float),
                'flexion_sign': float(coupling_flexion_sign),
            }
            all_variances.append(np.full(2, coupling_std**2))
            self.joint_noise_dim = 2

        elif self.joint_type == 'spring':
            if spring_axis_child is None or model_R_0 is None or spring_std is None:
                raise ValueError("For 'spring' joint, must provide spring_axis_child, "
                                 "model_R_0, and spring_std.")
            if spring_axis_child.shape != (3,):
                raise ValueError("spring_axis_child must have shape (3,)")
            if np.asarray(model_R_0).shape != (3, 3):
                raise ValueError("model_R_0 must have shape (3, 3)")

            self.joint_params = {
                'e_neutral': spring_axis_child / np.linalg.norm(spring_axis_child),
                'R_0': np.asarray(model_R_0, dtype=float),
            }
            all_variances.append(np.array([spring_std**2]))
            self.joint_noise_dim = 1

        elif self.joint_type is not None:
            raise ValueError(f"Unknown joint_type: '{self.joint_type}'. "
                             f"Must be one of {JOINT_TYPES}.")
        
        if self.num_vector_sensors == 0 and self.joint_type is None:
            raise ValueError("At least one vector sensor or a joint constraint must be provided.")
        
        self.R = np.diag(np.concatenate(all_variances))
        
        # --- Covariance and State Initialization ---
        self.P = np.eye(6) * init_orientation_std ** 2
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
            vector_sensor_data_c,
        )

        self.q_wp = q_lin_wp
        self.q_wc = q_lin_wc

    def set_qs(self, q_wp: Rotation, q_wc: Rotation):
        self.q_wp = q_wp
        self.q_wc = q_wc

    @staticmethod
    def _normalize_vector_measurements(vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Scales each vector sensor reading to unit length, keeping only its direction.

        A zero vector is passed through untouched rather than producing a NaN. This is
        not a numerical edge case but the normal path: mag_off zeroes the magnetometer
        for every sample and mag_adapt zeroes it on gated ones, and a zeroed sensor must
        stay zeroed so its residual and its H block both drop out of the update.

        THIS CHANGES THE SENSOR/GYRO TRUST RATIO, it is not a pure change of units. The
        residual h and the Jacobian H are both linear in the measurement, so scaling a
        reading by 1/|v| scales that sensor's block of both by 1/|v| — while R, built
        from the vector_sensor_stds at construction, does not move. In
        S = H P H^T + M R M^T the first term shrinks by 1/|v|^2 and the second does not,
        so the sensor is trusted LESS relative to the gyro prediction. For the
        accelerometer at |a| ~ 9.81 m/s^2 that is a factor of ~96 in variance: running
        normalized at acc_std s is equivalent to running unnormalized at acc_std ~9.81*s.
        The magnetometer is barely affected, since Xsens reports it in calibrated units
        where a nominal Earth field already reads ~1.0.

        So a normalized-vs-unnormalized comparison at fixed stds is a comparison of two
        different tunings, not of the geometry alone. To vary only the geometry, scale
        each sensor's std by its own nominal magnitude when enabling this.
        """
        normalized = []
        for v in vectors:
            norm = np.linalg.norm(v)
            normalized.append(v / norm if norm > 0.0 else v)
        return normalized

    def _heading_projection_axis(self, R_wp: np.ndarray, R_wc: np.ndarray,
                                 vector_sensor_data_p: List[np.ndarray],
                                 vector_sensor_data_c: List[np.ndarray]) -> Optional[np.ndarray]:
        """Unit world-frame direction the heading-only sensor's residual is confined to, or None.

        A relative rotation of theta about the reference direction a_w moves the restricted
        sensor's world reading by theta * (a_w_hat x v_w), so that cross product IS the only
        direction of the residual carrying information about the DOF the reference sensor
        cannot see. Projecting onto it keeps exactly that and discards the rest.

        The axis is built from the MEAN of the two segments' world readings, not from one of
        them: the residual is antisymmetric in parent and child, so choosing either side would
        make the retained direction depend on which segment was called the parent.

        Returns None when the projection is not defined -- either reading zero (which is how
        mag_off and mag_adapt switch the magnetometer off, and must stay a no-op rather than
        becoming a NaN), or the two directions closer than HEADING_AXIS_MIN_SINE to parallel.
        Both cases mean the sensor genuinely has nothing to say about the reference's null
        direction, so dropping its row is the answer rather than a fallback.
        """
        i = self.heading_only_sensor
        a_w = 0.5 * (R_wp @ vector_sensor_data_p[0] + R_wc @ vector_sensor_data_c[0])
        v_w = 0.5 * (R_wp @ vector_sensor_data_p[i] + R_wc @ vector_sensor_data_c[i])
        axis = np.cross(a_w, v_w)
        scale = np.linalg.norm(a_w) * np.linalg.norm(v_w)
        norm = np.linalg.norm(axis)
        if scale == 0.0 or norm < HEADING_AXIS_MIN_SINE * scale:
            return None
        return axis / norm

    def _heading_only_readings(self, R_wp: np.ndarray, R_wc: np.ndarray,
                               vector_sensor_data_p: List[np.ndarray],
                               vector_sensor_data_c: List[np.ndarray]
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """The restricted sensor's two readings, with the reference component stripped if asked.

        Both bodies strip against ONE shared world direction -- the mean of the reference
        sensor's two world-frame readings, rotated into each body frame -- and not each against
        its own reading. That distinction is the whole reason this lives in the filter instead of
        in a preprocessing pass over the data.

        With each body using its own reference, the two stripped readings are perpendicular to
        DIFFERENT world directions, so their difference acquires a component along the reference
        whenever the two reference sensors disagree, and the measurement manufactures a
        magnetometer residual out of an accelerometer disagreement -- 12% of the field magnitude
        at 30 deg of accelerometer disagreement, which is within this dataset's range at the
        ankle. With one shared direction the difference is perpendicular to it identically, for
        any state and any readings.
        """
        i = self.heading_only_sensor
        v_p, v_c = vector_sensor_data_p[i], vector_sensor_data_c[i]
        if not self.heading_only_pure:
            return v_p, v_c
        a_w = 0.5 * (R_wp @ vector_sensor_data_p[0] + R_wc @ vector_sensor_data_c[0])
        norm = np.linalg.norm(a_w)
        if norm == 0.0:
            return v_p, v_c
        a_hat = a_w / norm
        a_p, a_c = R_wp.T @ a_hat, R_wc.T @ a_hat
        return v_p - a_p * (a_p @ v_p), v_c - a_c * (a_c @ v_c)

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        m = np.zeros((3, 3))
        m[0, 1], m[0, 2] = -v[2], v[1]
        m[1, 0], m[1, 2] = v[2], -v[0]
        m[2, 0], m[2, 1] = -v[1], v[0]
        return m

    def _get_time_update(self, gyro_p: np.ndarray, gyro_c: np.ndarray, dt: float) -> Tuple[Rotation, Rotation]:
        """Predicts the next state based on gyroscope data."""
        R_p_update = Rotation.from_rotvec(dt * gyro_p).as_matrix()
        R_c_update = Rotation.from_rotvec(dt * gyro_c).as_matrix()
        
        P11 = self.P[:3, :3]
        P12 = self.P[:3, 3:]
        P22 = self.P[3:, 3:]
        
        new_P11 = R_p_update @ P11 @ R_p_update.T
        new_P12 = R_p_update @ P12 @ R_c_update.T
        new_P22 = R_c_update @ P22 @ R_c_update.T
        
        self.P[:3, :3] = new_P11
        self.P[:3, 3:] = new_P12
        self.P[3:, :3] = new_P12.T
        self.P[3:, 3:] = new_P22
        
        self.P += (dt**2) * self.Q

        q_lin_wp = self._get_gyro_orientation_estimate(self.q_wp, gyro_p, dt)
        q_lin_wc = self._get_gyro_orientation_estimate(self.q_wc, gyro_c, dt)
        return q_lin_wp, q_lin_wc

    def _get_gyro_orientation_estimate(self, q: Rotation, gyro: np.ndarray, dt: float) -> Rotation:
        """Integrates gyroscope readings to estimate orientation."""
        delta_q = Rotation.from_rotvec(dt * gyro)
        return q * delta_q

    def _get_measurement_update(self, q_lin_wp: Rotation, q_lin_wc: Rotation,
                                vector_sensor_data_p: List[np.ndarray],
                                vector_sensor_data_c: List[np.ndarray]) -> Tuple[Rotation, Rotation]:
        """Corrects the state prediction using sensor measurements."""

        if self.normalize_measurements:
            vector_sensor_data_p = self._normalize_vector_measurements(vector_sensor_data_p)
            vector_sensor_data_c = self._normalize_vector_measurements(vector_sensor_data_c)

        R_wp = q_lin_wp.as_matrix()
        R_wc = q_lin_wc.as_matrix()

        # Get Jacobians and residual, evaluated at eta = 0
        H = self.get_H_jacobian(R_wp, R_wc, vector_sensor_data_p, vector_sensor_data_c)
        e = self.get_h(R_wp, R_wc, vector_sensor_data_p, vector_sensor_data_c)
        
        M = self.get_M_jacobian(R_wp, R_wc)
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
        J_p = q_delta_p.as_matrix()
        J_c = q_delta_c.as_matrix()
        
        Pt11 = P_tilde[:3, :3]
        Pt12 = P_tilde[:3, 3:]
        Pt22 = P_tilde[3:, 3:]
        
        self.P[:3, :3] = J_p @ Pt11 @ J_p.T
        self.P[:3, 3:] = J_p @ Pt12 @ J_c.T
        self.P[3:, :3] = self.P[:3, 3:].T
        self.P[3:, 3:] = J_c @ Pt22 @ J_c.T
        
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
            v_p, v_c = vector_sensor_data_p[i], vector_sensor_data_c[i]
            if i == self.heading_only_sensor:
                v_p, v_c = self._heading_only_readings(
                    R_wp, R_wc, vector_sensor_data_p, vector_sensor_data_c)
            H_vec_p = R_wp @ self.skew_symmetric(v_p).T
            H_vec_c = -R_wc @ self.skew_symmetric(v_c).T
            H_block = np.hstack([H_vec_p, H_vec_c])
            if i == self.heading_only_sensor:
                # Same rank-1 projection get_h applies, and it has to be the same axis:
                # H must stay the exact derivative of h or the update is inconsistent.
                axis = self._heading_projection_axis(
                    R_wp, R_wc, vector_sensor_data_p, vector_sensor_data_c)
                H_block = (np.zeros((3, 6)) if axis is None
                           else np.outer(axis, axis @ H_block))
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
            # This is (dh_joint2 / d_eta), from paper Eq 7 & 9.
            #
            #   h = (R_wp v_J).(R_wc u_K) - cos(alpha)
            #
            # Perturbing on the RIGHT, R_wp -> R_wp exp([eta_p]), sends the parent's world-frame
            # axis to v_w + R_wp (eta_p x v_J), so
            #
            #   dh = [R_wp (eta_p x v_J)] . u_w = eta_p . (v_J x R_wp^T u_w)
            #                                   = eta_p . R_wp^T (v_w x u_w)
            #
            # using R(a x b) = (Ra) x (Rb). The child block is the same with the roles swapped,
            # which flips the sign of the cross product.
            #
            # BOTH FACTORS MATTER AND BOTH WERE MISSING. This used to be the bare world-frame
            # cross products np.cross(u_w, v_w) and np.cross(v_w, u_w) — no R^T, and negated.
            # The error state is a body-frame perturbation, so a world-frame gradient is not
            # the derivative with respect to it; the sign then sent the correction the wrong
            # way. Nothing caught it because the only test asserted H.shape == (7, 6), and
            # nothing in experiments/ has ever used a joint constraint. Numerical
            # differentiation of get_h is what found it, and test_jacobians_match_numerical
            # is what keeps it found.
            v_j = self.joint_params['v_j']
            u_k = self.joint_params['u_k']

            v_j_world = R_wp @ v_j
            u_k_world = R_wc @ u_k
            cross_world = np.cross(v_j_world, u_k_world)

            dh_d_eta_p = R_wp.T @ cross_world
            dh_d_eta_c = -R_wc.T @ cross_world

            H_block = np.hstack([dh_d_eta_p, dh_d_eta_c]).reshape(1, 6)
            all_H_blocks.append(H_block)

        elif self.joint_type == 'coupling':
            # r  = P^T nu - M^T c(q),   q = nu . u_c
            # dr/dnu = P^T - (M^T c'(q)) u_c^T          (2, 3)
            # dnu/deta from model_frame_jacobian        (3, 6)
            nu, _, derivative = self._coupling_state(R_wp, R_wc)
            channel_slope = self.joint_params['channel_map'].T @ derivative
            dr_dnu = (self.joint_params['basis_P'].T
                      - np.outer(channel_slope, self.joint_params['u_c']))
            all_H_blocks.append(dr_dnu @ joint_constraints.model_frame_jacobian(
                nu, self.joint_params['R_0']))

        elif self.joint_type == 'spring':
            nu = joint_constraints.relative_rotvec_in_model_frame(
                R_wp, R_wc, self.joint_params['R_0'])
            all_H_blocks.append(
                (self.joint_params['e_neutral']
                 @ joint_constraints.model_frame_jacobian(
                     nu, self.joint_params['R_0'])).reshape(1, 6))

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
            v_p, v_c = vector_sensor_data_p[i], vector_sensor_data_c[i]
            if i == self.heading_only_sensor:
                v_p, v_c = self._heading_only_readings(
                    R_wp, R_wc, vector_sensor_data_p, vector_sensor_data_c)
            p_world_frame = R_wp @ v_p
            c_world_frame = R_wc @ v_c
            h_block = p_world_frame - c_world_frame
            if i == self.heading_only_sensor:
                # Kept at 3 rows, rank 1, rather than collapsed to the scalar w . h, so that
                # the measurement layout -- and the compiled kernel's MEAS_DIM -- is unchanged.
                #
                # It is not an approximation of the 1-row filter, it IS the 1-row filter: the
                # two discarded rows carry residual 0 and Jacobian 0, so they drop out of
                # n = -K e and of K H, and their block of M R M^T is 2 sigma^2 I with no
                # off-diagonal coupling, which makes S block diagonal between them and
                # everything else. S^-1 restricted to the live rows is then exactly the inverse
                # the 4-row filter would have formed. Checked numerically against a hand-built
                # 3-acc-rows-plus-one-scalar filter: identical to 1e-21.
                #
                # THAT EXACTNESS NEEDS THE PER-AXIS MAG STDS TO BE EQUAL, which is what the
                # pipeline passes (np.ones(3) * mag_std). With anisotropic stds
                # R_wp diag(var) R_wp^T is no longer a multiple of the identity, the discarded
                # rows couple to the retained one through S, and the two forms part company at
                # the same order as the correction itself. Nothing here relies on it, but a
                # caller doing per-axis magnetometer noise should know it is no longer reading
                # the clean scalar measurement.
                #
                # Their diagonal of M R M^T is also what keeps S invertible; projecting M as
                # well would zero that block and the Cholesky solve would fail every sample.
                axis = self._heading_projection_axis(
                    R_wp, R_wc, vector_sensor_data_p, vector_sensor_data_c)
                h_block = np.zeros(3) if axis is None else axis * axis.dot(h_block)
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
            # Paper Eq 4: the angle between the two axes is fixed at alpha, so their
            # world-frame dot product is cos(alpha) and the residual is the shortfall.
            v_j = self.joint_params['v_j']
            u_k = self.joint_params['u_k']
            cos_alpha = self.joint_params['cos_alpha']

            v_j_world = R_wp @ v_j
            u_k_world = R_wc @ u_k

            h_scalar = v_j_world.dot(u_k_world) - cos_alpha
            all_h_blocks.append(np.array([h_scalar]))

        elif self.joint_type == 'coupling':
            # The two non-sagittal channels against the published curve. nu's component along
            # u_c IS the flexion angle by gauge, so the angle is read off the current estimate
            # rather than being a state or a free parameter.
            nu, q, _ = self._coupling_state(R_wp, R_wc)
            curve, _ = joint_constraints.reuben_design(
                np.array([q]), self.joint_params['flexion_sign'])
            target = self.joint_params['channel_map'].T @ curve[0]
            all_h_blocks.append(self.joint_params['basis_P'].T @ nu - target)

        elif self.joint_type == 'spring':
            nu = joint_constraints.relative_rotvec_in_model_frame(
                R_wp, R_wc, self.joint_params['R_0'])
            all_h_blocks.append(np.array([self.joint_params['e_neutral'].dot(nu)]))

        if not all_h_blocks:
            return np.empty((0,))
        return np.concatenate(all_h_blocks)

    def _coupling_state(self, R_wp: np.ndarray, R_wc: np.ndarray
                        ) -> Tuple[np.ndarray, float, np.ndarray]:
        """(nu, flexion angle q, dc/dq) for the coupling constraint at the current estimate.

        Shared by `get_h` and `get_H_jacobian` so the residual and its Jacobian are guaranteed
        to be evaluated at the same q. They are not independent: q is a function of the state,
        so a residual computed at one q and a Jacobian at another is not a linearization of
        anything, and the error would be invisible in any test that checked only shapes.
        """
        nu = joint_constraints.relative_rotvec_in_model_frame(
            R_wp, R_wc, self.joint_params['R_0'])
        q = float(nu @ self.joint_params['u_c'])
        _, derivative = joint_constraints.reuben_design(
            np.array([q]), self.joint_params['flexion_sign'])
        return nu, q, derivative[0]

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
        if self.joint_type is not None:
            # Every constraint's noise is additive on its own residual, so the block is the
            # identity at whatever dimension that residual has: 3 for the hinge's e_link,
            # 2 for the coupling's two channels, 1 for the universal's dot product and for
            # the spring's single channel.
            M[num_meas_vec:, num_noise_vec:] = np.eye(self.joint_noise_dim)
            
        # Off-diagonal blocks (dh_vec / d_nu_joint) and (dh_joint / d_nu_vec)
        # are zero, as their noises are independent.
        
        return M