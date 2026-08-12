"""
Covers experiments/acceleration_projection.py — the analysis that validates the rigid-body
acceleration projection against mocap.

Scope: the REFERENCE this analysis compares against, and its own numerical helpers. The
projection itself is not retested here — IMUTrace.project_acc is covered by
test/TestIMUTrace.py (invertibility, the zero-offset no-op, and the closed form for an offset
orthogonal to the rotation axis), and WorldTrace.get_joint_center by test/TestWorldTrace.py.

The reference is what this module adds, and it is the part that can go wrong invisibly. Built
with a transposed rotation, a flipped gravity sign, or the other segment's joint center, it
still has the right shape and a plausible magnitude — the analysis would simply report a
larger projection error, which reads as a finding rather than as a bug. So the tests below
check the reference against input whose answer is known in closed form (a stationary plate,
and a joint that is rigid by construction) rather than checking shapes.

Synthetic throughout — no dataset needed.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.acceleration_projection import (LOWPASS_CUTOFF_HZ, alignment_table,
                                                angle_between_deg, joint_center_reference_acc,
                                                joint_samples, joint_signals, lowpass,
                                                residual_alignment, spectra_table, traces_table)
from experiments.experiment_utils import EXPECTED_GRAVITY, _compute_perfect_joint_acc
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

FS = 100.0
DURATION = 8.0  # must clear 2 x TRIM_S with enough left over to measure

PARENT_OFFSET = np.array([0.03, -0.19, 0.02])  # sensor -> joint center, parent side
CHILD_OFFSET = np.array([-0.01, 0.16, 0.04])   # and child side, pointing back up the limb


def _smooth_curve(timestamps: np.ndarray, amplitudes, frequencies, phases) -> np.ndarray:
    """Sum of a few sinusoids per axis — smooth, non-degenerate, and band-limited well below
    LOWPASS_CUTOFF_HZ so that the analysis filter is not what any tolerance here measures."""
    return np.column_stack([
        sum(a * np.sin(2 * np.pi * f * timestamps + p) for a, f, p in zip(amp, freq, phase))
        for amp, freq, phase in zip(amplitudes, frequencies, phases)
    ])


def make_rigid_joint(fs: float = FS, duration: float = DURATION):
    """A synthetic joint that is exactly rigid, as (parent_plate, child_plate).

    Both segments are built BACKWARD from a shared joint-center path: given the joint center
    p_jc(t) and each segment's orientation R(t), the segment's sensor sits at
    p_jc(t) - R(t) @ offset, so the segment's own implied joint center R(t) @ offset + position
    is p_jc(t) exactly, for both segments and every sample. That is what makes this a fair
    test of the reference: the two segments cannot disagree about where the joint is, so the
    two reference points the module computes must coincide.

    Each segment's IMUTrace comes from WorldTrace.calculate_imu_trace with EXPECTED_GRAVITY,
    i.e. the specific force at the SENSOR, which is what a real accelerometer reports and what
    the projection starts from. No noise: noise would make the tolerances a statement about
    the noise level rather than about the construction.
    """
    n = int(duration * fs)
    timestamps = np.arange(n) / fs

    joint_center = _smooth_curve(timestamps,
                                 amplitudes=[(0.25, 0.05), (0.10, 0.03), (0.18, 0.04)],
                                 frequencies=[(0.7, 1.9), (1.1, 2.3), (0.5, 1.7)],
                                 phases=[(0.0, 1.2), (0.6, 2.1), (1.9, 0.4)])

    plates = []
    rotation_specs = [
        # Distinct frequencies per segment and per axis, so the joint-center least squares is
        # well conditioned: a segment that only ever rotates about one axis leaves the offset
        # along that axis unobservable and get_joint_center would return something arbitrary.
        dict(amplitudes=[(0.45, 0.12), (0.30, 0.09), (0.25, 0.07)],
             frequencies=[(0.6, 1.6), (0.9, 2.1), (0.4, 1.3)],
             phases=[(0.3, 1.1), (1.4, 0.2), (2.2, 0.8)]),
        dict(amplitudes=[(0.60, 0.15), (0.20, 0.08), (0.35, 0.10)],
             frequencies=[(0.8, 1.4), (1.3, 2.5), (0.5, 1.8)],
             phases=[(1.0, 0.5), (0.2, 1.7), (0.9, 2.4)]),
    ]
    for (name, offset, spec) in zip(('femur_r_imu', 'tibia_r_imu'),
                                    (PARENT_OFFSET, CHILD_OFFSET), rotation_specs):
        rotations = Rotation.from_rotvec(_smooth_curve(timestamps, **spec)).as_matrix()
        positions = joint_center - np.einsum('nij,j->ni', rotations, offset)
        world_trace = WorldTrace(timestamps, positions, rotations)
        imu_trace = world_trace.calculate_imu_trace(acc_from_gravity=EXPECTED_GRAVITY)
        plates.append(PlateTrial(name=name, imu_trace=imu_trace, world_trace=world_trace))
    return plates[0], plates[1]


class TestReferenceConstruction(unittest.TestCase):
    """joint_center_reference_acc — the mocap truth every error metric is measured against."""

    def test_stationary_plate_reads_gravity_in_its_own_frame(self):
        """The sharpest available check of the gravity convention, and the one that needs no
        differentiation at all: a plate that never moves must produce EXPECTED_GRAVITY rotated
        into its body frame, whatever the offset is. A flipped sign or a transposed rotation
        both fail here; both survive a magnitude check, which is why this compares vectors."""
        n = 200
        rotation = Rotation.from_euler('xyz', [30.0, -50.0, 15.0], degrees=True)
        rotations = np.repeat(rotation.as_matrix()[None], n, axis=0)
        world_trace = WorldTrace(np.arange(n) / FS, np.zeros((n, 3)), rotations)
        plate = PlateTrial('femur_r_imu', world_trace.calculate_imu_trace(EXPECTED_GRAVITY),
                           world_trace)

        body_acc, linear_acc = joint_center_reference_acc(plate, PARENT_OFFSET)
        expected = rotation.as_matrix().T @ EXPECTED_GRAVITY
        np.testing.assert_allclose(body_acc, np.broadcast_to(expected, (n, 3)), atol=1e-9)
        np.testing.assert_allclose(linear_acc, 0.0, atol=1e-9)

    def test_linear_acceleration_return_is_gravity_removed_in_the_world_frame(self):
        """The second return has to be the first with gravity taken back out, in the world
        frame. Panel scaling and the ref_lin_norm column both read it as a linear
        acceleration, so a body-frame subtraction here would silently smear gravity across
        all three axes."""
        parent, _ = make_rigid_joint()
        body_acc, linear_acc = joint_center_reference_acc(parent, PARENT_OFFSET)
        world_acc = np.einsum('nij,nj->ni', parent.world_trace.rotations, body_acc)
        np.testing.assert_allclose(world_acc - EXPECTED_GRAVITY, linear_acc, atol=1e-9)

    def test_own_and_shared_joint_centers_agree_when_the_joint_is_rigid(self):
        """joint_center_reference_acc (the segment's OWN implied joint center) and
        experiment_utils._compute_perfect_joint_acc (the MIDPOINT of both segments') are
        different reference points on real data, and the module reports both. On a joint that
        is rigid by construction they are the same point, so they must agree — which is what
        pins the local construction as mirroring the pipeline's rather than merely resembling
        it. Any difference in frame convention, gravity sign or differentiation order between
        the two shows up here as a mismatch."""
        parent, child = make_rigid_joint()
        parent_offset, child_offset, _ = parent.world_trace.get_joint_center(child.world_trace)
        shared = dict(zip(('parent', 'child'), _compute_perfect_joint_acc(parent, child)))
        for role, plate, offset in (('parent', parent, parent_offset),
                                    ('child', child, child_offset)):
            own, _ = joint_center_reference_acc(plate, offset)
            np.testing.assert_allclose(own, shared[role], atol=1e-9)


class TestTrialTables(unittest.TestCase):
    """The per-trial tables the figures read, on the rigid synthetic joint."""

    @classmethod
    def setUpClass(cls):
        cls.parent, cls.child = make_rigid_joint()
        cls.signals = joint_signals(cls.parent, cls.child, FS)
        cls.samples = joint_samples({'R_Knee': cls.signals})
        cls.alignment = alignment_table({'R_Knee': cls.signals})

    def test_fixture_offsets_are_the_ones_the_joint_was_built_from(self):
        """Precondition for everything below, not coverage of get_joint_center (that is
        TestWorldTrace's): if the offsets came back wrong, these tables would be describing
        a different point than the one the fixture placed."""
        for role, expected in (('parent', PARENT_OFFSET), ('child', CHILD_OFFSET)):
            np.testing.assert_allclose(self.signals[role]['offset'], expected, atol=1e-6)

    def test_correction_is_the_difference_between_the_two_estimates(self):
        """corr_norm has to be |a_proj - a_sensor| and not, say, |a_proj - a_true|: panel E
        reads it as the amount of correction applied, and the identity line drawn on that
        panel is only the ideal if this holds."""
        expected = np.linalg.norm(
            self.signals['parent']['projected'] - self.signals['parent']['sensor'], axis=1)
        stored = self.samples.loc[self.samples['role'] == 'parent', 'corr_norm'].to_numpy()
        np.testing.assert_allclose(stored, expected[:len(expected):3][:len(stored)],
                                   rtol=1e-4, atol=1e-4)

    def test_error_columns_are_measured_against_the_reference(self):
        """err_proj and err_raw both have to use the same reference, or the comparison the
        whole figure rests on is between two different questions."""
        for column, signal in (('err_proj', 'projected'), ('err_raw', 'sensor')):
            expected = np.linalg.norm(
                self.signals['parent'][signal] - self.signals['parent']['reference'], axis=1)
            stored = self.samples.loc[self.samples['role'] == 'parent', column].to_numpy()
            np.testing.assert_allclose(stored, expected[:len(expected):3][:len(stored)],
                                       rtol=1e-4, atol=1e-4)

    def test_sample_table_is_finite(self):
        """Every metric column has to be finite. angle_between_deg is the one that can quietly
        produce NaN (arccos of a dot product a hair outside [-1, 1]), and these two signals are
        near-parallel almost everywhere, so that is the ordinary case here, not a corner one."""
        numeric = self.samples.select_dtypes(include=[np.floating])
        bad = numeric.columns[~np.isfinite(numeric).all()].tolist()
        self.assertFalse(bad, f"non-finite values in {bad}")

    def test_alignment_table_reports_no_floor_on_perfectly_aligned_data(self):
        """The synthetic sensor frames ARE the segment frames, so the fitted residual rotation
        must come out near zero. On real data that angle is the sensor-to-segment
        misalignment; here there is none, so a non-trivial angle would mean the Kabsch fit is
        absorbing something it should not."""
        self.assertTrue((self.alignment['align_angle_deg'] < 0.5).all())
        self.assertTrue((self.alignment['r2_proj'] > 0.99).all())
        self.assertTrue((self.alignment['r2_proj'] > self.alignment['r2_raw']).all())

    def test_secondary_tables_are_shaped_as_the_figures_expect(self):
        traces = traces_table({'R_Knee': self.signals}, joint='R_Knee')
        self.assertEqual(set(traces['role']), {'parent', 'child'})
        for prefix in ('ref', 'proj', 'sens'):
            for axis in 'xyz':
                self.assertIn(f'{prefix}_{axis}', traces.columns)

        spectra = spectra_table({'R_Knee': self.signals}, FS)
        self.assertLessEqual(spectra['freq_hz'].max(), FS / 2)
        for column in ('psd_reference', 'psd_projected', 'psd_sensor'):
            self.assertTrue((spectra[column] >= 0).all())


class TestSignalHelpers(unittest.TestCase):
    """The numerical helpers, each on input with a closed-form answer."""

    def test_residual_alignment_recovers_a_known_rotation(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(500, 3)) + EXPECTED_GRAVITY
        true_rotation = Rotation.from_euler('xyz', [1.5, -0.8, 0.4], degrees=True)
        estimate = true_rotation.inv().apply(reference)

        _, angle_deg, rms_before, rms_after = residual_alignment(estimate, reference)
        self.assertAlmostEqual(angle_deg, np.degrees(true_rotation.magnitude()), places=4)
        self.assertGreater(rms_before, 0.1)
        self.assertLess(rms_after, 1e-9)

    def test_residual_alignment_is_a_rotation_not_a_reflection(self):
        """Kabsch without the determinant term returns a reflection whenever that fits
        better, which is not a frame misalignment and would report a meaningless angle."""
        rng = np.random.default_rng(1)
        reference = rng.normal(size=(200, 3))
        mirrored = reference * np.array([1.0, 1.0, -1.0])
        rotation, _, _, _ = residual_alignment(mirrored, reference)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_angle_between_deg_on_known_pairs(self):
        vectors = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]])
        others = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [1.0, 0, 0]])
        np.testing.assert_allclose(angle_between_deg(vectors, others), [0.0, 90.0, 180.0, 45.0],
                                   atol=1e-6)

    def test_angle_between_deg_survives_identical_vectors(self):
        """Identical vectors give a normalized dot product that can land just above 1.0 in
        floating point; unclipped, arccos returns NaN. Finiteness is the assertion — the
        residual angle is arccos's own precision near zero, ~1e-6 deg, not an error."""
        rng = np.random.default_rng(2)
        vectors = rng.normal(size=(1000, 3)) * 9.81
        angles = angle_between_deg(vectors, vectors.copy())
        self.assertTrue(np.isfinite(angles).all())
        np.testing.assert_allclose(angles, 0.0, atol=1e-4)

    def test_lowpass_keeps_the_signal_and_removes_the_noise_band(self):
        """A 1 Hz component has to survive intact and a 40 Hz one has to be gone — the 40 Hz
        band is where the twice-differentiated marker trace lives, and the point of filtering
        is that it never reaches the error metrics."""
        timestamps = np.arange(2000) / FS
        low = np.sin(2 * np.pi * 1.0 * timestamps)
        high = 0.5 * np.sin(2 * np.pi * 40.0 * timestamps)
        signal = np.column_stack([low + high] * 3)

        filtered = lowpass(signal, FS)
        interior = slice(200, -200)  # filtfilt edges are not what this is testing
        np.testing.assert_allclose(filtered[interior, 0], low[interior], atol=0.02)

    def test_lowpass_is_zero_lag(self):
        """filtfilt rather than lfilter: a lag between the reference and the estimate would be
        counted as error rather than as lag. Checked by comparing a well-in-band sine to its
        filtered self — one sample of lag on a 2 Hz sine would displace it by ~0.13, two
        orders of magnitude above the tolerance below."""
        timestamps = np.arange(1000) / FS
        signal = np.column_stack([np.sin(2 * np.pi * 2.0 * timestamps)] * 3)
        filtered = lowpass(signal, FS)
        interior = slice(150, -150)
        self.assertLess(np.max(np.abs(filtered[interior] - signal[interior])), 0.01)

    def test_lowpass_cutoff_is_below_nyquist(self):
        """Guards the constant itself: butter raises for a cutoff above Nyquist, but a cutoff
        just under it would silently do nothing."""
        self.assertLess(LOWPASS_CUTOFF_HZ, FS / 4)


if __name__ == '__main__':
    unittest.main()
