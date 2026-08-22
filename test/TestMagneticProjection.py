"""
Covers experiments/magnetic_projection.py — the analysis that asks whether a magnetometer reading
transports to the joint centre, and whether an array of magnetometers can estimate the field
gradient and do better.

WHAT THESE TESTS ARE FOR. Every quantity in that module is a difference between two estimates of
one magnetic field, and every way it can be wrong produces a plausible number rather than a crash:

  * a transposed rotation. R carries a body-frame reading into the world; R^T carries it back.
    Either direction produces a world-frame "field" of the right magnitude that rotates with the
    limb, which reads as a spatial disturbance.
  * a gradient fitted with the separation and the difference the wrong way round, or with G
    transposed. Both return a tensor of about the right size, and only the physical constraint
    (symmetric, traceless) or a known-answer fixture catches it.
  * a mechanism decomposition that attributes a hard iron to a field gradient, which is the exact
    conclusion the experiment exists to rule in or out — and the unconstrained 9-parameter
    gradient model CAN partly mimic a body-fixed constant, so this is not hypothetical.
  * a magnitude channel that quietly uses mocap after all, which would destroy the one claim this
    family makes that the accelerometer version cannot.
  * an angular column filled with a degenerate zero, which reads as perfect agreement.

So the fixtures below have answers known in CLOSED FORM. Three worlds are built over the same
kinematics — a perfectly uniform field, a field with a known constant gradient, and a uniform
field seen through known hard irons — and each one has a different right answer that the module
must produce. On the uniform world every residual is exactly zero; on the gradient world the
gradient fits must recover the tensor that was injected; on the hard-iron world the mechanism
decomposition must name the sensor, not the room. Tolerances are numerical rather than
statistical, so a convention error fails by orders of magnitude rather than by a few percent.

The windowing, conditioning and signal helpers are NOT retested here: they are imported from
experiments/acceleration_projection.py and covered by test/TestAccelerationProjection.py. What
this module adds is the field models, the projection modes, the calibration and the mechanism
decomposition, and that is what is exercised.

Synthetic throughout — no dataset needed.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
from scipy.spatial.transform import Rotation

import experiments.magnetic_projection as mp
from experiments.magnetic_projection import (CALIBRATIONS, MODES, PAIR_CLASSES,
                                             PRIMARY_CALIBRATION,
                                             PRIMARY_MODE, SegmentLine, TrialContext,
                                             _SYMMETRIC_TRACELESS_BASIS, apply_calibration,
                                             as_magnitude, fit_calibration, fit_gradient,
                                             covariation_rows, fit_local_field_and_bias,
                                             instantaneous_gradient, joint_comparisons,
                                             pair_class,
                                             lab_field_map, marker_comparisons, mechanism_rows,
                                             sample_rows, segment_comparisons, segment_sensors,
                                             stats_row, trial_gradient)
from experiments.experiment_utils import EXPECTED_GRAVITY
from experiments.global_assumptions import DatasetSpec
from experiments.joint_center import MIN_FIT_FRAMES, joint_offsets
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

FS = 100.0
# Long enough that a run clears 2 x the window trim (3 periods of the 6 Hz cutoff at each end, so
# 100 samples total at 100 Hz) with seconds left, AND clears joint_center's MIN_FIT_FRAMES.
DURATION = 12.0

# The uniform world field, in the arbitrary units the Xsens export uses (|B| ~ 1). Deliberately
# not axis-aligned: a field along one axis would let a transposed rotation pass on two of three
# components.
FIELD = np.array([0.20, -0.76, 0.42])

# A KNOWN, PHYSICALLY ADMISSIBLE gradient: symmetric and traceless, as curl B = 0 and div B = 0
# require. Built from the module's own basis so the test cannot pass by agreeing with a
# differently-parametrized copy of it, and scaled to ~0.3 a.u./m, the order the lab field map
# measures on real data.
GRADIENT = np.einsum('k,kij->ij', np.array([0.12, -0.07, 0.09, 0.05, -0.11]),
                     _SYMMETRIC_TRACELESS_BASIS)

# Per-sensor hard irons for the third world. Distinct in direction and size, and large enough
# (~10% of the field) to dominate everything else, matching what the real data shows.
HARD_IRONS = {
    'THIGH_R_H': np.array([0.05, -0.02, 0.03]),
    'THIGH_R_M': np.array([-0.03, 0.06, 0.01]),
    'THIGH_R_L': np.array([0.02, 0.04, -0.05]),
    'SHANK_R_H': np.array([-0.06, 0.01, 0.02]),
    'SHANK_R_M': np.array([0.01, -0.05, -0.03]),
    'SHANK_R_L': np.array([0.04, 0.03, 0.05]),
    'PELVIS_M': np.array([-0.02, -0.01, 0.04]),
}

# Sensor positions in their segment's own frame, three per limb segment. Spread ~170 mm along the
# segment with 6 mm off the line, which is the geometry IMoVE actually has (measured: 160-193 mm
# along, 5-8 mm off) — and it is the geometry, not the field, that decides whether a segment-scale
# fit can work, so a fixture that made the array straighter or wider than it is would test a
# different question.
SEGMENT_SITES = {
    'H': np.array([0.004, 0.085, -0.002]),
    'M': np.array([-0.002, 0.000, 0.003]),
    'L': np.array([0.005, -0.085, -0.001]),
}
# Each sensor's own body-to-segment rotation. Distinct and non-trivial, because a fixture with
# identical sensor frames would pass with every frame transform omitted.
MISALIGNMENTS = {
    'H': Rotation.from_euler('xyz', [4.0, -7.0, 11.0], degrees=True).as_matrix(),
    'M': Rotation.from_euler('xyz', [-9.0, 3.0, -5.0], degrees=True).as_matrix(),
    'L': Rotation.from_euler('xyz', [6.0, 12.0, -3.0], degrees=True).as_matrix(),
}

# Where each joint centre sits in the proximal segment's frame, and where the distal segment's
# Mid sensor sits relative to it. Long enough that the joint centre is a real extrapolation
# beyond the sensor array, which is the geometric fact the experiment turns on.
KNEE_IN_THIGH = np.array([0.01, -0.24, 0.02])
KNEE_IN_SHANK = np.array([-0.01, 0.20, 0.01])
HIP_IN_PELVIS = np.array([0.09, -0.08, 0.01])
HIP_IN_THIGH = np.array([0.00, 0.22, 0.00])

ROTATION_SPECS = (
    dict(amplitudes=[(0.45, 0.12), (0.30, 0.09), (0.25, 0.07)],
         frequencies=[(0.6, 1.6), (0.9, 2.1), (0.4, 1.3)],
         phases=[(0.3, 1.1), (1.4, 0.2), (2.2, 0.8)]),
    dict(amplitudes=[(0.60, 0.15), (0.20, 0.08), (0.35, 0.10)],
         frequencies=[(0.8, 1.4), (1.3, 2.5), (0.5, 1.8)],
         phases=[(1.0, 0.5), (0.2, 1.7), (0.9, 2.4)]),
    dict(amplitudes=[(0.22, 0.06), (0.40, 0.11), (0.18, 0.05)],
         frequencies=[(0.5, 1.1), (0.7, 1.9), (1.2, 2.2)],
         phases=[(0.7, 1.9), (2.0, 0.4), (0.1, 1.5)]),
)
PATH_SPEC = dict(amplitudes=[(0.25, 0.05), (0.10, 0.03), (0.18, 0.04)],
                 frequencies=[(0.7, 1.9), (1.1, 2.3), (0.5, 1.7)],
                 phases=[(0.0, 1.2), (0.6, 2.1), (1.9, 0.4)])

FIELD_ORIGIN = np.array([0.0, 1.0, 0.0])


def _smooth_curve(timestamps: np.ndarray, amplitudes, frequencies, phases) -> np.ndarray:
    """Sum of a few sinusoids per axis — smooth, non-degenerate, and band-limited well below the
    6 Hz analysis cutoff so that the filter is not what any tolerance here measures."""
    return np.column_stack([
        sum(a * np.sin(2 * np.pi * f * timestamps + p) for a, f, p in zip(amp, freq, phase))
        for amp, freq, phase in zip(amplitudes, frequencies, phases)
    ])


# Amplitude of the shared, time-varying field used by the covariation fixture. Every sensor sees
# the SAME wobble at the same instant, which is what a room-scale disturbance looks like to a body
# walking through it — so the right answer for every pair is perfect correlation.
WOBBLE = 0.06
WOBBLE_HZ = 0.3


def world_field(positions: np.ndarray, gradient=None,
                timestamps: np.ndarray = None) -> np.ndarray:
    """The synthetic world-frame field at a set of world positions.

    `timestamps` adds a common time-varying term — identical at every sensor, so it is a purely
    SHARED disturbance and every pair must come back perfectly correlated.
    """
    if gradient is None:
        field = np.tile(FIELD, (len(positions), 1))
    else:
        field = FIELD + (np.asarray(positions) - FIELD_ORIGIN) @ np.asarray(gradient).T
    if timestamps is not None:
        phase = 2 * np.pi * WOBBLE_HZ * timestamps
        field = field + WOBBLE * np.column_stack([np.sin(phase), np.cos(phase),
                                                  np.sin(phase + 1.0)])
    return field


def make_plates(gradient=None, hard_irons=None, fs: float = FS, duration: float = DURATION,
                wobble: bool = False):
    """Seven plates over three rigid segments, with the magnetometers synthesized exactly.

    Built the way the real datasets are: one pose per SEGMENT, several sensors at fixed positions
    in that segment's frame, each carrying its own body-to-segment rotation, and each plate's
    world trace holding its own IMU position and its own sensor->world rotation — which is what
    `imove_mocap.load_trial` produces after `shift_world_origin`.

    The magnetometer is then m_i(t) = R_i(t)^T B(x_i(t)) + b_i, exactly. So the world-frame field
    the module reconstructs, R_i m_i, is the injected field plus the injected hard iron rotated
    into the world, and every quantity downstream has a closed form.

    Both joints are rigid by construction — the distal segment is placed FROM the joint centre the
    proximal one implies — so no cross-segment residual here can come from the joint model.
    """
    n = int(duration * fs)
    timestamps = np.arange(n) / fs
    hard_irons = hard_irons or {}

    pelvis_position = _smooth_curve(timestamps, **PATH_SPEC) + np.array([0.0, 1.0, 0.0])
    poses = {'PELVIS': (pelvis_position,
                        Rotation.from_rotvec(_smooth_curve(timestamps,
                                                           **ROTATION_SPECS[2])).as_matrix())}
    thigh_rotation = Rotation.from_rotvec(_smooth_curve(timestamps, **ROTATION_SPECS[0])).as_matrix()
    hip = poses['PELVIS'][0] + np.einsum('nij,j->ni', poses['PELVIS'][1], HIP_IN_PELVIS)
    poses['THIGH_R'] = (hip - np.einsum('nij,j->ni', thigh_rotation, HIP_IN_THIGH), thigh_rotation)

    shank_rotation = Rotation.from_rotvec(_smooth_curve(timestamps, **ROTATION_SPECS[1])).as_matrix()
    knee = poses['THIGH_R'][0] + np.einsum('nij,j->ni', poses['THIGH_R'][1], KNEE_IN_THIGH)
    poses['SHANK_R'] = (knee - np.einsum('nij,j->ni', shank_rotation, KNEE_IN_SHANK),
                        shank_rotation)

    plates = {}
    layout = {'PELVIS': ('M',), 'THIGH_R': ('H', 'M', 'L'), 'SHANK_R': ('H', 'M', 'L')}
    for segment, placements in layout.items():
        position, rotation = poses[segment]
        for placement in placements:
            name = f'{segment}_{placement}'
            site = SEGMENT_SITES[placement]
            sensor_position = position + np.einsum('nij,j->ni', rotation, site)
            sensor_rotation = np.einsum('nij,jk->nik', rotation, MISALIGNMENTS[placement])
            world_trace = WorldTrace(timestamps, sensor_position, sensor_rotation)
            imu_trace = world_trace.calculate_imu_trace(EXPECTED_GRAVITY)
            field = world_field(sensor_position, gradient,
                                timestamps if wobble else None)
            imu_trace.mag = (np.einsum('nji,nj->ni', sensor_rotation, field)
                             + hard_irons.get(name, np.zeros(3)))
            plates[name] = PlateTrial(name=name, imu_trace=imu_trace, world_trace=world_trace)
    return plates


TEST_SPEC = DatasetSpec(
    name='synthetic',
    segment_sensor={'Pelvis': 'PELVIS_M',
                    'Thigh R High': 'THIGH_R_H', 'Thigh R Mid': 'THIGH_R_M',
                    'Thigh R Low': 'THIGH_R_L',
                    'Shank R High': 'SHANK_R_H', 'Shank R Mid': 'SHANK_R_M',
                    'Shank R Low': 'SHANK_R_L'},
    joints={'R_Hip': ('PELVIS_M', 'THIGH_R_M'), 'R_Knee': ('THIGH_R_M', 'SHANK_R_M')},
    primary_joints=('R_Hip', 'R_Knee'),
    pelvis_sensor='PELVIS_M',
    foot_sensors=(),
    subject_label='{}',
    field_reference={},
)


def make_context(gradient=None, hard_irons=None, calibration: str = PRIMARY_CALIBRATION,
                 parameters=None, wobble: bool = False) -> TrialContext:
    plates = make_plates(gradient, hard_irons, wobble=wobble)
    offsets = joint_offsets(plates, TEST_SPEC, min_frames=MIN_FIT_FRAMES)
    return TrialContext(plates, TEST_SPEC, 'synthetic', 's1', 't1', parameters or {},
                        calibration, offsets)


def stats_frame(comparisons):
    import pandas as pd
    return pd.DataFrame([stats_row(c) for c in comparisons])


# ==============================================================================
# World 1: a perfectly uniform field
# ==============================================================================

class TestUniformField(unittest.TestCase):
    """With one constant field everywhere, EVERY residual in the experiment must be zero.

    This is the test that catches a transposed rotation, a wrong reference point, a mode that
    projects along the wrong displacement, and a held-out prediction that reads the sensor it is
    supposed to be predicting. None of those can survive a world where the right answer is the
    same vector at every point.
    """

    @classmethod
    def setUpClass(cls):
        cls.context = make_context()

    def test_the_world_frame_field_is_the_injected_constant(self):
        for name, field in self.context.B.items():
            np.testing.assert_allclose(field, np.tile(FIELD, (len(field), 1)), atol=1e-10,
                                       err_msg=f"{name} does not reconstruct the injected field")

    def test_every_cross_segment_comparison_is_exact(self):
        frame = stats_frame(joint_comparisons(self.context))
        self.assertFalse(frame.empty)
        vector = frame[frame['channel'] == 'vector']
        self.assertLess(vector['err_p50'].max(), 1e-8)
        self.assertLess(vector['ang_p50'].max(), 1e-4)

    def test_every_mode_is_exact_when_the_field_is_uniform(self):
        """Including the gradient modes: a fit over a uniform field must return G = 0, so every
        mode's correction is zero and every mode reproduces the 0th order exactly."""
        frame = stats_frame(joint_comparisons(self.context))
        modes = set(frame['mode'])
        self.assertGreater(len(modes), 4, "the fixture should exercise the multi-sensor modes")
        for mode in modes:
            worst = frame.loc[(frame['mode'] == mode) & (frame['channel'] == 'vector'),
                              'err_p50'].max()
            self.assertLess(worst, 1e-6, f"mode {mode} is not exact on a uniform field")

    def test_same_segment_comparisons_are_exact(self):
        frame = stats_frame(segment_comparisons(self.context))
        self.assertFalse(frame.empty)
        self.assertLess(frame['err_p50'].max(), 1e-6)

    def test_the_held_out_sensor_is_predicted_exactly(self):
        frame = stats_frame(segment_comparisons(self.context))
        held_out = frame[frame['target_kind'] == 'held_out']
        # Two sensors are left after holding one out, so a constant and a line can be fitted and a
        # quadratic cannot. That the quadratic is ABSENT rather than silently fitted through two
        # points is the property being pinned.
        self.assertEqual(set(held_out['mode']), {'mean', 'linear'})
        self.assertLess(held_out['err_p50'].max(), 1e-6)

    def test_the_marker_family_matches_the_reference_it_was_given(self):
        frame = stats_frame(marker_comparisons(self.context, FIELD, 1))
        self.assertFalse(frame.empty)
        self.assertLess(frame['err_p50'].max(), 1e-6)

    def test_the_lab_map_is_perfect_at_every_order(self):
        """Scored on the RESIDUAL, not on r2. A uniform field has no variance to explain, so r2 is
        a ratio of two quantities that are both zero to float precision and means nothing here —
        which is itself worth pinning, because reading r2 off a near-uniform field is how a map
        gets credited with skill it does not have."""
        for order in (0, 1, 2):
            summary = lab_field_map(self.context.B, self.context.X, order, stride=5)
            self.assertLess(summary['residual_loso_p50'], 1e-6,
                            f"order {order} leaves a residual on a uniform field")
            self.assertLess(summary['gradient_norm'], 1e-6,
                            f"order {order} invented a gradient in a uniform field")

    def test_the_fitted_gradient_is_zero(self):
        _, G, _ = instantaneous_gradient(self.context.B, self.context.X, self.context.names)
        self.assertLess(np.abs(G).max(), 1e-6)
        G_trial, _ = trial_gradient(self.context.B, self.context.X, self.context.names)
        self.assertLess(np.abs(G_trial).max(), 1e-6)


# ==============================================================================
# World 2: a known, physically admissible gradient
# ==============================================================================

class TestKnownGradient(unittest.TestCase):
    """With a known constant gradient injected, the fits must recover THAT TENSOR.

    Recovering the right magnitude is not enough — a transposed G or a fit that swapped the
    regressor and the target both return something of the right size. These compare the matrix.
    """

    @classmethod
    def setUpClass(cls):
        cls.context = make_context(gradient=GRADIENT)

    def test_instantaneous_fit_recovers_the_tensor_and_the_ambient_field(self):
        B0, G, centre = instantaneous_gradient(self.context.B, self.context.X,
                                               self.context.names)
        np.testing.assert_allclose(G, np.tile(GRADIENT, (len(G), 1, 1)), atol=1e-6)
        expected = world_field(centre, GRADIENT)
        np.testing.assert_allclose(B0, expected, atol=1e-6)

    def test_trial_fit_recovers_the_tensor(self):
        G, explained = trial_gradient(self.context.B, self.context.X, self.context.names)
        np.testing.assert_allclose(G, GRADIENT, atol=1e-6)
        self.assertGreater(explained, 0.999)

    def test_the_physical_constraint_costs_nothing_on_a_real_gradient(self):
        """THE TEST THE EXPERIMENT'S HEADLINE CLAIM RESTS ON. A genuine magnetostatic gradient is
        symmetric and traceless, so constraining the fit to five parameters must lose nothing.
        Only if that holds here does the constraint costing a lot on real data mean anything."""
        G, explained = trial_gradient(self.context.B, self.context.X, self.context.names)
        G_physical, explained_physical = trial_gradient(self.context.B, self.context.X,
                                                        self.context.names, physical=True)
        np.testing.assert_allclose(G_physical, GRADIENT, atol=1e-6)
        self.assertAlmostEqual(explained, explained_physical, places=6)

    def test_the_lab_map_sees_the_gradient_and_reports_its_size(self):
        summary = lab_field_map(self.context.B, self.context.X, 1, stride=5)
        self.assertGreater(summary['r2_loso'], 0.999)
        self.assertAlmostEqual(summary['gradient_norm'], float(np.linalg.norm(GRADIENT)),
                               places=6)

    def test_the_0th_order_is_wrong_by_the_gradient_over_the_lever_arm(self):
        """The size of the error the projection is supposed to fix, in closed form: transporting
        a reading over a displacement d in a field with gradient G is wrong by exactly G d."""
        frame = stats_frame(joint_comparisons(self.context))
        zeroth = frame[(frame['mode'] == PRIMARY_MODE) & (frame['channel'] == 'vector')]
        self.assertFalse(zeroth.empty)
        for _, row in zeroth.iterrows():
            separation = float(row['sep_m'])
            self.assertGreater(row['err_p50'], 0.1 * np.linalg.norm(GRADIENT) * separation)

    def test_the_gradient_modes_remove_it(self):
        frame = stats_frame(joint_comparisons(self.context))
        vector = frame[frame['channel'] == 'vector']
        zeroth = vector.loc[vector['mode'] == PRIMARY_MODE, 'err_p50'].median()
        for mode in ('trial_gradient', 'trial_gradient_phys', 'body_linear'):
            corrected = vector.loc[vector['mode'] == mode, 'err_p50']
            if corrected.empty:
                continue
            self.assertLess(corrected.max(), 1e-6,
                            f"{mode} should be exact when the field really is linear")
        self.assertGreater(zeroth, 1e-4, "the fixture must leave the 0th order something to fix")

    def test_the_mechanism_names_the_gradient(self):
        rows = mechanism_rows(self.context)
        self.assertTrue(rows)
        for row in rows:
            self.assertGreater(row['explained_gradient'], 0.999)
            self.assertGreater(row['explained_gradient_physical'], 0.999)
            self.assertAlmostEqual(row['gradient_norm_physical'],
                                   float(np.linalg.norm(GRADIENT)), places=5)


# ==============================================================================
# World 3: uniform field, known hard irons
# ==============================================================================

class TestHardIron(unittest.TestCase):
    """A per-sensor offset must be recovered as a per-sensor offset, not as a field gradient.

    This is the discrimination the whole experiment turns on, and the failure mode is silent: a
    nine-parameter gradient fitted to a body-fixed difference returns a tensor, and that tensor
    looks like a finding.
    """

    @classmethod
    def setUpClass(cls):
        cls.context = make_context(hard_irons=HARD_IRONS)

    def test_the_calibration_recovers_the_injected_offset(self):
        for name, plate in self.context.plates.items():
            fit = fit_calibration(plate.imu_trace.mag, plate.world_trace.rotations, FIELD,
                                  np.asarray(plate.valid))
            recovered = np.array([fit['hard_iron_x'], fit['hard_iron_y'], fit['hard_iron_z']])
            np.testing.assert_allclose(recovered, HARD_IRONS[name], atol=1e-8,
                                       err_msg=f"{name}'s hard iron was not recovered")
            self.assertLess(fit['residual_hard_iron'], 1e-8)
            self.assertGreater(fit['residual_raw'], 1e-3)

    def test_applying_the_calibration_restores_the_uniform_field(self):
        parameters = {name: {'hard_iron': HARD_IRONS[name]} for name in HARD_IRONS}
        for name, plate in self.context.plates.items():
            corrected, source = apply_calibration(np.asarray(plate.imu_trace.mag),
                                                  parameters[name], 'hard_iron')
            self.assertEqual(source, 'other')
            world = np.einsum('nij,nj->ni', plate.world_trace.rotations, corrected)
            np.testing.assert_allclose(world, np.tile(FIELD, (len(world), 1)), atol=1e-8)

    def test_a_missing_fit_leaves_the_reading_alone_and_says_so(self):
        corrected, source = apply_calibration(np.zeros((5, 3)), {}, 'hard_iron')
        self.assertEqual(source, 'none')
        np.testing.assert_array_equal(corrected, np.zeros((5, 3)))

    def test_the_mechanism_names_the_sensor_not_the_field(self):
        rows = mechanism_rows(self.context)
        self.assertTrue(rows)
        for row in rows:
            self.assertGreater(row['explained_body'], 0.999,
                               "a body-fixed offset must be explained by the body-fixed model")
            self.assertLess(row['explained_gradient_physical'], 0.9,
                            "a hard iron is not a physically admissible field gradient")

    def test_the_body_constant_is_the_difference_of_the_two_hard_irons(self):
        """In closed form: with a uniform field, B_a - B_b = R_a b_a - R_b b_b, and expressing it
        in sensor A's frame gives b_a - C_ab b_b, a constant. That constant is what the body model
        must recover."""
        rows = {(row['sensor_a'], row['sensor_b']): row for row in mechanism_rows(self.context)}
        for (a, b), row in rows.items():
            transform = MISALIGNMENTS[a[-1]].T @ MISALIGNMENTS[b[-1]]
            expected = HARD_IRONS[a] - transform @ HARD_IRONS[b]
            self.assertAlmostEqual(row['body_constant_norm'], float(np.linalg.norm(expected)),
                                   places=6)

    def test_a_hard_iron_survives_every_spatial_mode(self):
        """No projection can remove it, because it is not a property of position. If a mode
        appears to, that mode is fitting the sensors."""
        frame = stats_frame(joint_comparisons(self.context))
        vector = frame[(frame['channel'] == 'vector') & (frame['mode'] != 'body_model')]
        self.assertGreater(vector['err_p50'].min(), 1e-4)


# ==============================================================================
# Geometry
# ==============================================================================

class TestSegmentGeometry(unittest.TestCase):
    """SegmentLine — the arc-length parametrization every segment-scale mode is built on."""

    @classmethod
    def setUpClass(cls):
        cls.context = make_context()
        cls.line = cls.context.line('THIGH_R')

    def test_sensors_are_ordered_along_the_array(self):
        """Monotone along the array, in whichever direction the principal axis came out.

        The DIRECTION is arbitrary — an SVD's leading singular vector has no preferred sign — and
        nothing downstream depends on it, since the arc-length coordinate is signed and every fit
        is a polynomial in it. What does matter is that the middle sensor is in the middle: the
        order sets which two sensors define the axis, and a scrambled order would define it
        between two adjacent sensors instead of across the whole array.
        """
        self.assertIn(list(self.line.order),
                      [['THIGH_R_L', 'THIGH_R_M', 'THIGH_R_H'],
                       ['THIGH_R_H', 'THIGH_R_M', 'THIGH_R_L']])

    def test_the_array_geometry_is_measured_in_the_body_frame(self):
        """The span and the off-line spread are body-frame facts, and the segment rotates through
        ~50 deg in this fixture — averaging world-frame displacements would shrink both."""
        sites = np.array([SEGMENT_SITES[p] for p in 'HML'])
        direction = np.linalg.svd(sites - sites.mean(axis=0), full_matrices=False)[2][0]
        along = sites @ direction
        self.assertAlmostEqual(self.line.along_mm, 1000 * (along.max() - along.min()), places=3)
        perpendicular = sites - np.outer(along, direction)
        expected = 1000 * np.max(np.linalg.norm(perpendicular - perpendicular.mean(axis=0),
                                                axis=1))
        self.assertAlmostEqual(self.line.offline_mm, expected, places=3)

    def test_the_span_matches_the_fixture(self):
        expected = np.linalg.norm(SEGMENT_SITES['H'] - SEGMENT_SITES['L'])
        self.assertAlmostEqual(self.line.span, expected, places=6)

    def test_extrapolation_is_zero_inside_the_array_and_grows_outside(self):
        inside = self.line.coordinate(self.context.X['THIGH_R_M'])[0]
        self.assertEqual(self.line.extrapolation_ratio(inside), 0.0)
        knee = self.context.joint_centre('R_Knee')
        outside, offline = self.line.coordinate(knee)
        self.assertGreater(self.line.extrapolation_ratio(outside), 0.3,
                           "the knee should be a real extrapolation beyond the thigh array")
        self.assertGreater(float(np.median(offline)), 0.0)

    def test_evaluating_at_a_sensor_returns_that_sensor_reading(self):
        """A polynomial that interpolates its own nodes: evaluating the quadratic at one of the
        three sensors must return that sensor's field exactly, whatever the field is."""
        context = make_context(gradient=GRADIENT)
        line = context.line('THIGH_R')
        for name in line.names:
            estimate = line.evaluate(context.X[name], 2)
            np.testing.assert_allclose(estimate, context.B[name], atol=1e-6)

    def test_segments_are_grouped_off_the_placement_suffix(self):
        groups = segment_sensors(['THIGH_R_H', 'THIGH_R_M', 'PELVIS_M', 'calcn_r_imu'])
        self.assertEqual(groups['THIGH_R'], ['THIGH_R_H', 'THIGH_R_M'])
        self.assertEqual(groups['PELVIS'], ['PELVIS_M'])
        self.assertEqual(groups['calcn_r_imu'], ['calcn_r_imu'])


# ==============================================================================
# Channels, controls and bookkeeping
# ==============================================================================

class TestChannelsAndControls(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = make_context(hard_irons=HARD_IRONS)
        cls.comparisons = joint_comparisons(cls.context)

    def test_as_magnitude_carries_the_norm_and_nothing_else(self):
        vectors = np.array([[3.0, 4.0, 0.0], [0.0, -1.0, 0.0]])
        magnitudes = as_magnitude(vectors)
        np.testing.assert_allclose(magnitudes[:, 0], [5.0, 1.0])
        np.testing.assert_allclose(magnitudes[:, 1:], 0.0)

    def test_the_norm_channel_never_touches_a_rotation(self):
        """The claim that distinguishes this family from the accelerometer version: |R m| = |m|,
        so the magnitude comparison is identical whatever the mocap says. Re-running it against
        plates whose rotations have been replaced by garbage must change nothing."""
        norm_rows = [c for c in self.comparisons if c.channel == 'norm']
        self.assertTrue(norm_rows)
        before = [float(np.median(np.linalg.norm(c.estimate - c.truth, axis=1)))
                  for c in norm_rows]

        broken = make_plates(hard_irons=HARD_IRONS)
        random = Rotation.random(len(next(iter(broken.values()))), random_state=0).as_matrix()
        for plate in broken.values():
            plate.world_trace.rotations = random
        offsets = joint_offsets(self.context.plates, TEST_SPEC, min_frames=MIN_FIT_FRAMES)
        context = TrialContext(broken, TEST_SPEC, 'synthetic', 's1', 't1', {},
                               PRIMARY_CALIBRATION, offsets)
        after = [float(np.median(np.linalg.norm(c.estimate - c.truth, axis=1)))
                 for c in joint_comparisons(context) if c.channel == 'norm']
        np.testing.assert_allclose(sorted(after), sorted(before), atol=1e-10)

    def test_the_norm_channel_blanks_what_it_cannot_measure(self):
        for comparison in self.comparisons:
            if comparison.channel != 'norm':
                continue
            row = stats_row(comparison)
            self.assertTrue(np.isnan(row['ang_p50']), "a magnitude gap has no direction")
            self.assertTrue(np.isnan(row['align_angle_deg']))
            frame = sample_rows(comparison)
            self.assertTrue(frame['ang_proj'].isna().all())

    def test_the_norm_channel_is_scored_over_the_whole_record(self):
        scopes = {c.channel: c.scope for c in self.comparisons}
        self.assertEqual(scopes.get('norm'), 'full')
        self.assertEqual(scopes.get('vector'), 'mocap')

    def test_the_control_mode_agrees_perfectly_and_says_nothing(self):
        """`body_model` must score exactly zero disagreement — that is what makes it the control —
        while sitting further from the reference than the 0th order. If it ever stops scoring
        zero, it has stopped being the degenerate arm the report warns about."""
        frame = stats_frame(self.comparisons)
        control = frame[(frame['mode'] == 'body_model') & (frame['channel'] == 'vector')]
        self.assertFalse(control.empty)
        self.assertLess(control['err_p50'].max(), 1e-10)

        marker = stats_frame(marker_comparisons(self.context, FIELD, 1))
        control_reference = marker.loc[marker['mode'] == 'body_model', 'err_p50'].median()
        zeroth_reference = marker.loc[marker['mode'] == PRIMARY_MODE, 'err_p50'].median()
        self.assertGreater(control_reference, zeroth_reference,
                           "the control buys its perfect agreement by moving away from the truth")

    def test_the_body_fit_excludes_the_pair_it_will_be_scored_on(self):
        full = self.context.body_fit(())
        excluded = self.context.body_fit(('THIGH_R_M', 'SHANK_R_M'))
        self.assertEqual(full['n_sensors'], len(self.context.names))
        self.assertEqual(excluded['n_sensors'], len(self.context.names) - 2)

    def test_the_body_fit_is_unavailable_when_too_few_sensors_remain(self):
        self.assertIsNone(self.context.body_fit(tuple(self.context.names[:-2])))

    def test_every_row_records_which_calibration_produced_it(self):
        for calibration in CALIBRATIONS:
            context = make_context(hard_irons=HARD_IRONS, calibration=calibration)
            row = stats_row(joint_comparisons(context)[0])
            self.assertEqual(row['calibration'], calibration)
            # Nothing was supplied, so every arm falls back to the uncorrected reading and must
            # say so rather than claiming a calibration it did not apply.
            self.assertEqual(row['calibration_source'], 'none')


class TestPhysicalGradientBasis(unittest.TestCase):
    """The five-parameter basis, which is what makes the physical constraint a constraint."""

    def test_the_basis_is_symmetric_and_traceless(self):
        for basis in _SYMMETRIC_TRACELESS_BASIS:
            np.testing.assert_allclose(basis, basis.T)
            self.assertAlmostEqual(float(np.trace(basis)), 0.0)

    def test_the_basis_is_independent_and_spans_five_dimensions(self):
        flattened = _SYMMETRIC_TRACELESS_BASIS.reshape(5, 9)
        self.assertEqual(np.linalg.matrix_rank(flattened), 5)

    def test_a_constrained_fit_returns_a_symmetric_traceless_tensor(self):
        rng = np.random.default_rng(0)
        separation = rng.normal(size=(500, 3))
        # A difference that is NOT a physical gradient: an antisymmetric tensor, which curl B = 0
        # forbids. The constrained fit must refuse to reproduce it.
        antisymmetric = np.array([[0.0, 0.3, -0.2], [-0.3, 0.0, 0.1], [0.2, -0.1, 0.0]])
        difference = separation @ antisymmetric.T
        G, explained = fit_gradient(separation, difference, physical=True)
        np.testing.assert_allclose(G, G.T, atol=1e-9)
        self.assertAlmostEqual(float(np.trace(G)), 0.0, places=9)
        self.assertLess(explained, 0.05)
        _, unconstrained = fit_gradient(separation, difference)
        self.assertGreater(unconstrained, 0.999)

    def test_an_unconstrained_fit_recovers_a_transposed_tensor_correctly(self):
        """G is applied as `separation @ G.T`, so a transposed return would still explain the data
        for a symmetric tensor and would fail for an asymmetric one. Checked with an asymmetric
        tensor for exactly that reason."""
        rng = np.random.default_rng(1)
        separation = rng.normal(size=(400, 3))
        tensor = np.array([[0.1, 0.4, -0.2], [0.0, -0.3, 0.5], [0.2, 0.1, 0.2]])
        G, explained = fit_gradient(separation, separation @ tensor.T)
        np.testing.assert_allclose(G, tensor, atol=1e-9)
        self.assertGreater(explained, 0.999)


# ==============================================================================
# Covariation
# ==============================================================================

class TestCovariation(unittest.TestCase):
    """How much of what two sensors see is SHARED — the question that decides whether a relative
    filter's subtraction removes a disturbance or manufactures one.

    Two worlds with opposite right answers: one where every sensor sees the SAME time-varying
    disturbance (perfectly shared, so differencing must remove all of it) and one where the only
    fluctuation is each sensor's own bias rotating (private, so differencing must not help)."""

    def test_pair_class_reads_adjacency_off_the_spec(self):
        context = make_context()
        self.assertEqual(pair_class(context, 'THIGH_R_H', 'THIGH_R_L'), 'same_segment')
        self.assertEqual(pair_class(context, 'THIGH_R_M', 'SHANK_R_M'), 'across_joint')
        self.assertEqual(pair_class(context, 'THIGH_R_H', 'SHANK_R_L'), 'same_limb')
        # The side token sits in a different place in each dataset's naming — IMoVE's 'THIGH_R'
        # against Al Borno's 'femur_r_imu' — and keying on the wrong one collapses every pair into
        # 'distant', which silently removes the control the comparison rests on.
        self.assertEqual(mp._side_and_stem('THIGH_R'), ('r', 'THIGH'))
        self.assertEqual(mp._side_and_stem('femur_r_imu'), ('r', 'femur_imu'))
        self.assertEqual(mp._side_and_stem('torso_imu'), ('', 'torso_imu'))
        self.assertEqual(mp._side_and_stem('PELVIS'), ('', 'PELVIS'))
        # Every label the classifier can emit has to be one the report and the figures know.
        for a in context.names:
            for b in context.names:
                if a != b:
                    self.assertIn(pair_class(context, a, b), PAIR_CLASSES)

    def test_a_shared_disturbance_is_perfectly_correlated_and_cancels(self):
        """Every sensor sees the same wobble, so rho must be 1 and NOTHING may survive the
        difference. This is the case a relative filter is betting on."""
        rows = covariation_rows(make_context(wobble=True))
        self.assertTrue(rows)
        for row in rows:
            self.assertGreater(row['rho'], 0.999,
                               f"{row['sensor_a']}-{row['sensor_b']} should see one shared field")
            self.assertLess(row['surviving'], 1e-4,
                            "a perfectly shared disturbance must cancel in the difference")

    def test_a_private_disturbance_does_not_cancel(self):
        """With a static field and independent per-sensor biases, the only world-frame fluctuation
        is each sensor's own bias sweeping as it rotates. Two sensors on ONE segment rotate
        together so theirs stay partly correlated, but nothing may look perfectly shared, and
        differencing must not remove it all."""
        rows = covariation_rows(make_context(hard_irons=HARD_IRONS))
        self.assertTrue(rows)
        for row in rows:
            self.assertLess(row['rho'], 0.99)
            self.assertGreater(row['surviving'], 1e-3)

    def test_removing_the_bias_recovers_the_shared_disturbance(self):
        """The panel the figure turns on. With a shared wobble AND independent biases, the raw
        correlation is dragged down by the biases; subtracting each sensor's own fitted bias must
        put it back to ~1 and drive the surviving fraction to ~0."""
        rows = covariation_rows(make_context(hard_irons=HARD_IRONS, wobble=True))
        self.assertTrue(rows)
        raw = np.median([row['rho'] for row in rows])
        debiased = np.median([row['rho_debiased'] for row in rows])
        self.assertLess(raw, 0.99, "the independent biases should decorrelate the pair")
        self.assertGreater(debiased, 0.99, "removing them should recover the shared field")
        self.assertLess(np.median([row['surviving_debiased'] for row in rows]), 0.01)

    def test_the_joint_estimator_recovers_field_and_bias_without_a_reference(self):
        """`fit_local_field_and_bias` assumes no reference field, so it must return both the
        injected field and the injected bias from the readings and rotations alone."""
        plates = make_plates(hard_irons=HARD_IRONS)
        for name, plate in plates.items():
            valid = np.asarray(plate.valid)
            field, bias, condition, weakest = fit_local_field_and_bias(
                np.asarray(plate.imu_trace.mag)[valid],
                np.asarray(plate.world_trace.rotations)[valid])
            np.testing.assert_allclose(field, FIELD, atol=1e-6, err_msg=f'{name} field')
            np.testing.assert_allclose(bias, HARD_IRONS[name], atol=1e-6, err_msg=f'{name} bias')
            self.assertGreater(condition, 1.0)
            self.assertAlmostEqual(float(np.linalg.norm(weakest)), 1.0, places=6)

    def test_the_joint_estimator_reports_a_degenerate_trial_as_degenerate(self):
        """Rotation about ONE axis leaves the field's component along it constant in the body
        frame, hence indistinguishable from the bias along the same axis. The estimate must come
        back with a large condition number rather than a confident wrong answer — that number is
        what the analysis gates on."""
        n = int(DURATION * FS)
        timestamps = np.arange(n) / FS
        axis = np.array([0.0, 1.0, 0.0])
        rotations = Rotation.from_rotvec(np.outer(np.sin(2 * np.pi * 0.5 * timestamps),
                                                  axis)).as_matrix()
        mag = np.einsum('nji,j->ni', rotations, FIELD) + np.array([0.05, -0.02, 0.03])
        _, _, condition, _ = fit_local_field_and_bias(mag, rotations)
        self.assertGreater(condition, 1e6, "a single-axis trial cannot observe all of b")


if __name__ == '__main__':
    unittest.main()
