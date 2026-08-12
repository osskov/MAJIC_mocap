"""
Covers the magnetic-distortion dial (experiment_utils._compute_scaled_mag, _mag_override)
and the dose tables experiments/distortion_tolerance.py plots its x-axis from.

WHY THIS NEEDS PINNING. The sweep's entire claim to being trustworthy is that its two ends
are not new code paths: a scale of 0 must reproduce the existing mag oracle exactly and a
scale of 1 must reproduce the real magnetometer exactly, so a curve that lands on the
mag_on and perfect_mag arms at its ends is evidence the dial is right. Every way of getting
the interpolation wrong preserves that property at ONE end and breaks it at the other — a
transposed rotation still gives the oracle at 0, a sign flip still gives the real reading at
1 — and neither raises, neither changes the output's shape, and both leave a plausible
monotone curve in between. So the endpoints are tested as exact equalities, not tolerances.

The dose tables have the same shape of hazard: an inter-sensor field disagreement computed
in the body frame instead of the world frame would measure the two segments' relative
ORIENTATION and report tens of degrees of "distortion" in a perfectly uniform field. That
case is tested directly.

Synthetic throughout — no dataset needed.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from experiments.distortion_tolerance import (DISTORTION_SCALES, UNITY_SCALE, _angle_deg,
                                              check_unity_arm, distortion_method,
                                              distortion_tables, scaled_world_fields)
from experiments.experiment_utils import (_compute_expected_mag_field, _compute_perfect_mag,
                                          _compute_scaled_mag, _mag_override,
                                          compute_joint_angles)
from test.TestExperimentPhysics import WORLD_MAG, make_plate

# A field a distorted sensor reads instead of WORLD_MAG: different direction AND different
# magnitude, since a real anomaly changes both and a test that only turned the vector would
# pass for an implementation that dropped the magnitude term.
DISTORTED_MAG = np.array([-10.0, 25.0, 15.0])


def make_trial(fields: dict, n: int = 120) -> dict:
    """A trial's worth of plates, each rotating differently and each sitting in its own
    (constant, world-frame) magnetic field. `fields` maps sensor name -> world field.

    Different rotation rates per sensor are the point: any implementation that confuses the
    body and world frames gets the right answer when every sensor shares an orientation.
    """
    return {name: make_plate(name, n=n, omega=np.array([0.3, -0.7, 2.0]) * (i + 1),
                             world_mag=field)
            for i, (name, field) in enumerate(fields.items())}


class TestScaledMagEndpoints(unittest.TestCase):
    """The two exact equalities the sweep rests on."""

    def setUp(self):
        self.plate = make_plate('calcn_r_imu', world_mag=DISTORTED_MAG)

    def test_scale_one_reproduces_the_real_reading_exactly(self):
        """Not 'closely'. If a=1 is not the identity, every other scale is a fraction of
        something other than the measured distortion."""
        np.testing.assert_allclose(_compute_scaled_mag(self.plate, WORLD_MAG, 1.0),
                                   self.plate.imu_trace.mag, atol=1e-12)

    def test_scale_zero_reproduces_the_mag_oracle_exactly(self):
        np.testing.assert_allclose(_compute_scaled_mag(self.plate, WORLD_MAG, 0.0),
                                   _compute_perfect_mag(self.plate, WORLD_MAG), atol=1e-12)

    def test_the_endpoints_differ_for_a_distorted_sensor(self):
        """Guards the two tests above from passing vacuously: they would both hold for a
        sensor whose reading already equals the oracle."""
        gap = np.abs(_compute_scaled_mag(self.plate, WORLD_MAG, 1.0)
                     - _compute_scaled_mag(self.plate, WORLD_MAG, 0.0)).max()
        self.assertGreater(gap, 1.0)

    def test_an_undistorted_sensor_is_unmoved_by_any_scale(self):
        """A sensor already reading the assumed field has no distortion to scale, so the
        dial is a no-op on it at every setting — including the amplified ones."""
        clean = make_plate('torso_imu', world_mag=WORLD_MAG)
        for scale in (0.0, 0.5, 1.0, 2.0):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(_compute_scaled_mag(clean, WORLD_MAG, scale),
                                           clean.imu_trace.mag, atol=1e-12)


class TestScaledMagInterpolation(unittest.TestCase):
    def setUp(self):
        self.plate = make_plate('calcn_r_imu', world_mag=DISTORTED_MAG)
        self.rotations = self.plate.world_trace.rotations

    def _world_residual(self, scale: float) -> np.ndarray:
        body = _compute_scaled_mag(self.plate, WORLD_MAG, scale)
        world = np.einsum('nij,nj->ni', self.rotations, body)
        return world - WORLD_MAG

    def test_the_world_frame_residual_scales_by_exactly_the_scale(self):
        """The definition, checked in the frame it is defined in: the leftover against the
        assumed field is a * the measured leftover, sample by sample."""
        measured = self._world_residual(1.0)
        for scale in (0.0, 0.25, 0.5, 2.0):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(self._world_residual(scale), scale * measured,
                                           atol=1e-9)

    def test_scales_above_one_amplify_rather_than_saturate(self):
        """The sweep runs past 1.0 to find a breaking point this lab's field may not reach.
        If the dial clipped at 1, those arms would silently duplicate the real reading."""
        norms = [np.linalg.norm(self._world_residual(s), axis=1).mean()
                 for s in (1.0, 1.5, 2.0)]
        self.assertTrue(np.all(np.diff(norms) > 0.1), f"residual did not grow: {norms}")

    def test_it_is_linear_so_the_frame_choice_is_not_load_bearing(self):
        """Because the operation is linear and the frame change is a rotation, scaling the
        residual in the world frame is identical to interpolating between the oracle and the
        real reading in the BODY frame. Worth pinning: it means the implementation cannot
        hide a transpose error that only shows up in one frame, and it is why the endpoints
        can be exact."""
        oracle = _compute_perfect_mag(self.plate, WORLD_MAG)
        real = self.plate.imu_trace.mag
        for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(_compute_scaled_mag(self.plate, WORLD_MAG, scale),
                                           oracle + scale * (real - oracle), atol=1e-9)

    def test_the_magnitude_moves_with_the_scale(self):
        """Documented in _compute_scaled_mag and asserted here so it stays a known
        consequence rather than a surprise: this is an interpolation, not a rotation, so
        |m| is |e| at a=0 and the measured magnitude at a=1. The filter sees raw-magnitude
        measurements, so its magnetometer weighting drifts slightly across the sweep."""
        at_zero = np.linalg.norm(_compute_scaled_mag(self.plate, WORLD_MAG, 0.0), axis=1)
        at_one = np.linalg.norm(_compute_scaled_mag(self.plate, WORLD_MAG, 1.0), axis=1)
        np.testing.assert_allclose(at_zero, np.linalg.norm(WORLD_MAG), atol=1e-9)
        np.testing.assert_allclose(at_one, np.linalg.norm(DISTORTED_MAG), atol=1e-9)


class TestMagOverrideDispatch(unittest.TestCase):
    """The single place the three mag sources are resolved. Both the relative-filter and the
    EKF path go through it, so a source honoured on one and ignored on the other would run
    the requested configuration under one name and the default under another."""

    def setUp(self):
        self.plate = make_plate('calcn_r_imu', world_mag=DISTORTED_MAG)

    def test_real_leaves_the_reading_alone(self):
        self.assertIsNone(_mag_override(self.plate, 'real', None, None))

    def test_perfect_is_the_oracle(self):
        np.testing.assert_allclose(_mag_override(self.plate, 'perfect', WORLD_MAG, None),
                                   _compute_perfect_mag(self.plate, WORLD_MAG), atol=1e-12)

    def test_perfect_ignores_a_scale_that_was_passed_anyway(self):
        """The two cannot be requested together by name (resolve_method_spec raises), so if
        both arrive here the source is what was asked for."""
        np.testing.assert_allclose(_mag_override(self.plate, 'perfect', WORLD_MAG, 0.5),
                                   _compute_perfect_mag(self.plate, WORLD_MAG), atol=1e-12)

    def test_scaled_uses_the_scale(self):
        np.testing.assert_allclose(_mag_override(self.plate, 'scaled', WORLD_MAG, 0.5),
                                   _compute_scaled_mag(self.plate, WORLD_MAG, 0.5), atol=1e-12)

    def test_scaled_without_a_scale_raises_rather_than_defaulting(self):
        """There is no safe default: 0 would silently turn the arm into an oracle and 1 into
        the plain method, and both would look like a plausible result."""
        with self.assertRaises(ValueError):
            _mag_override(self.plate, 'scaled', WORLD_MAG, None)

    def test_an_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            _mag_override(self.plate, 'approximate', WORLD_MAG, 1.0)


class TestMethodNamesRunTheRightPhysics(unittest.TestCase):
    """End to end through compute_joint_angles, which is what the sweep actually calls. The
    name-parsing tests (TestMethodSpec) prove the spec is right; these prove the spec is
    obeyed all the way to the joint angles.

    Two plates only, so a single joint (Lumbar) runs and the filter pass stays fast.
    """

    @classmethod
    def setUpClass(cls):
        cls.plates = make_trial({'torso_imu': WORLD_MAG,        # the field reference
                                 'pelvis_imu': DISTORTED_MAG},  # the distorted partner
                                n=60)

    def _angles(self, method: str) -> np.ndarray:
        df = compute_joint_angles(self.plates, method)
        return df[['rx', 'ry', 'rz']].to_numpy()

    def test_the_field_reference_is_the_one_the_oracle_uses(self):
        """Sanity check on the fixture: torso_imu defines the expected field, so a scale of 0
        collapses the pelvis onto it."""
        np.testing.assert_allclose(_compute_expected_mag_field(list(self.plates.values())),
                                   WORLD_MAG, atol=1e-9)

    def test_unity_scale_reproduces_plain_mag_on(self):
        """The sweep's own wiring check, at the level of joint angles. This is the assertion
        check_unity_arm makes on real data."""
        np.testing.assert_allclose(self._angles(distortion_method(UNITY_SCALE)),
                                   self._angles('mag_on'), atol=1e-10)

    def test_zero_scale_reproduces_the_mag_oracle_arm(self):
        np.testing.assert_allclose(self._angles('mag_on_dist0.00'),
                                   self._angles('mag_on_real_acc_perfect_mag'), atol=1e-10)

    def test_an_intermediate_scale_is_a_different_estimate(self):
        """Guards the two above: if the distortion scale were ignored altogether, every arm
        would equal mag_on and both endpoint tests would still pass at a=1."""
        half = self._angles('mag_on_dist0.50')
        self.assertGreater(np.abs(half - self._angles('mag_on')).max(), 1e-6)
        self.assertGreater(np.abs(half - self._angles('mag_on_dist0.00')).max(), 1e-6)

    def test_mag_off_ignores_the_scale(self):
        """Documented, not endorsed, matching the threshold suffix's behaviour on a
        non-adapt base: mag_off zeroes the magnetometer after the override is applied, so a
        scale on it is mag_off recomputed under a second filename."""
        np.testing.assert_allclose(self._angles('mag_off_dist0.50'),
                                   self._angles('mag_off'), atol=1e-12)


class TestAngleHelper(unittest.TestCase):
    """_angle_deg, which every number on the figure's physical x-axis comes out of."""

    def test_known_angles(self):
        x, y = np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 2.0, 0.0]])
        self.assertAlmostEqual(float(_angle_deg(x, y)[0]), 90.0, places=9)
        self.assertAlmostEqual(float(_angle_deg(x, -x)[0]), 180.0, places=9)
        self.assertAlmostEqual(float(_angle_deg(x, 5.0 * x)[0]), 0.0, places=9)

    def test_it_is_exactly_zero_for_identical_vectors(self):
        """The a=0 arm's whole row of the dose table is this case. arccos of a normalized dot
        product would return a few hundredths of a degree of floating-point noise here, which
        would read as 'even the oracle sees some distortion'."""
        v = np.tile(DISTORTED_MAG, (50, 1))
        np.testing.assert_array_equal(_angle_deg(v, v.copy()), np.zeros(50))

    def test_it_is_insensitive_to_magnitude(self):
        """The dose axis is about direction; a magnitude error is reported separately as
        magdev. A scale-dependent angle would double-count it."""
        u = np.array([[3.0, -1.0, 2.0]])
        np.testing.assert_allclose(_angle_deg(u, np.array([[1.0, 1.0, 0.0]])),
                                   _angle_deg(100.0 * u, np.array([[0.01, 0.01, 0.0]])),
                                   atol=1e-9)


class TestDistortionTables(unittest.TestCase):
    """The dose tables: how much distortion a scale actually is, in degrees."""

    @classmethod
    def setUpClass(cls):
        # torso_imu is the only clean sensor, and it is also the one the field reference is
        # taken from, so Lumbar (pelvis, torso) is a MIXED joint — one distorted sensor
        # against the reference — while R_Ankle (tibia_r, calcn_r) has both its sensors in the
        # same distorted field. The two cases are what separate an inter-sensor disagreement
        # from an absolute field error, and the relative filter only pays for the former.
        cls.plates = make_trial({'torso_imu': WORLD_MAG, 'pelvis_imu': DISTORTED_MAG,
                                 'tibia_r_imu': DISTORTED_MAG, 'calcn_r_imu': DISTORTED_MAG},
                                n=80)
        cls.tables = distortion_tables('99', 'walking', cls.plates, DISTORTION_SCALES)
        cls.segment = cls.tables['segment_distortion']
        cls.joint = cls.tables['joint_distortion']

    def test_both_tables_cover_every_scale(self):
        for table in (self.segment, self.joint):
            np.testing.assert_allclose(np.sort(table['distortion_scale'].unique()),
                                       np.sort(np.asarray(DISTORTION_SCALES, dtype=float)))

    def test_only_joints_whose_sensors_are_present_are_listed(self):
        """Matches _joint_angles_from_filter, which skips those joints. A joint in the dose
        table with no accuracy row would silently drop out of every join downstream."""
        self.assertEqual(set(self.joint['joint']), {'Lumbar', 'R_Ankle'})

    def test_everything_is_zero_at_a_scale_of_zero(self):
        """At a=0 every sensor reads the same assumed field, so there is no distortion to
        report — absolute or relative."""
        for table, columns in ((self.segment, ['magdev_mean', 'field_angle_deg_mean',
                                               'field_angle_deg_p95']),
                               (self.joint, ['disagreement_deg_mean', 'disagreement_deg_p95'])):
            at_zero = table[table['distortion_scale'] == 0.0]
            for column in columns:
                with self.subTest(column=column):
                    np.testing.assert_allclose(at_zero[column].to_numpy(), 0.0, atol=1e-9)

    def test_a_clean_sensor_reports_no_distortion_at_any_scale(self):
        clean = self.segment[self.segment['sensor'] == 'torso_imu']
        np.testing.assert_allclose(clean['field_angle_deg_mean'].to_numpy(), 0.0, atol=1e-9)

    def test_the_segment_residual_is_linear_in_the_scale(self):
        """|a * d| = a * |d|, which is what makes the scale readable as a percentage of the
        measured distortion."""
        distorted = self.segment[self.segment['sensor'] == 'calcn_r_imu'].set_index('distortion_scale')
        unity = float(distorted.loc[UNITY_SCALE, 'magdev_mean'])
        for scale in DISTORTION_SCALES:
            with self.subTest(scale=scale):
                self.assertAlmostEqual(float(distorted.loc[scale, 'magdev_mean']),
                                       float(scale) * unity, places=6)

    def test_the_residual_fraction_is_relative_to_the_assumed_field(self):
        row = self.segment[(self.segment['sensor'] == 'calcn_r_imu')
                           & (self.segment['distortion_scale'] == UNITY_SCALE)].iloc[0]
        self.assertAlmostEqual(row['magdev_frac_mean'],
                               row['magdev_mean'] / np.linalg.norm(WORLD_MAG), places=9)

    def test_the_segment_angle_grows_with_the_scale(self):
        distorted = (self.segment[self.segment['sensor'] == 'calcn_r_imu']
                     .sort_values('distortion_scale')['field_angle_deg_mean'].to_numpy())
        self.assertTrue(np.all(np.diff(distorted) > 0), f"not monotone: {distorted}")

    def test_two_sensors_in_the_same_field_never_disagree(self):
        """THE frame test. R_Ankle's two sensors sit in one (distorted) field but rotate at
        different rates, so their body-frame readings differ by tens of degrees at every
        sample while the field they are in is identical. A disagreement computed in the body
        frame would report that orientation difference as distortion; the relative filter
        does not suffer from it, because a distortion common to both sensors cancels."""
        ankle = self.joint[self.joint['joint'] == 'R_Ankle']
        np.testing.assert_allclose(ankle['disagreement_deg_p95'].to_numpy(), 0.0, atol=1e-9)

    def test_a_mixed_joint_does_disagree_and_grows_with_the_scale(self):
        """Lumbar here spans one clean and one distorted sensor, which is the case the
        relative correction actually pays for."""
        lumbar = (self.joint[self.joint['joint'] == 'Lumbar']
                  .sort_values('distortion_scale')['disagreement_deg_mean'].to_numpy())
        self.assertTrue(np.all(np.diff(lumbar) > 0), f"not monotone: {lumbar}")
        self.assertGreater(lumbar[-1], 10.0)

    def test_the_joint_disagreement_matches_the_fields_it_is_computed_from(self):
        """Recomputed the long way round from scaled_world_fields, so the table's numbers are
        checked against the definition and not just against each other."""
        fields = scaled_world_fields(self.plates, WORLD_MAG, 0.5)
        expected = float(np.mean(_angle_deg(fields['pelvis_imu'], fields['torso_imu'])))
        row = self.joint[(self.joint['joint'] == 'Lumbar')
                         & (self.joint['distortion_scale'] == 0.5)].iloc[0]
        self.assertAlmostEqual(row['disagreement_deg_mean'], expected, places=9)

    def test_the_sample_count_is_recorded(self):
        self.assertTrue((self.segment['n_samples'] == 80).all())
        self.assertTrue((self.joint['n_samples'] == 80).all())


class TestUnityArmCheck(unittest.TestCase):
    """check_unity_arm, the sweep's self-audit. It has to complain when the two arms that are
    the same configuration by construction disagree, and stay quiet otherwise."""

    def _stats(self, unity_rmse: float, mag_on_rmse: float = 0.1) -> pd.DataFrame:
        rows = [{'subject': 'Subject01', 'trial_type': 'walking', 'joint_name': 'R_Knee',
                 'axis': 'MAG', 'method': method, 'rmse_rad': rmse}
                for method, rmse in ((distortion_method(UNITY_SCALE), unity_rmse),
                                     ('mag_on', mag_on_rmse), ('mag_off', 0.2))]
        return pd.DataFrame(rows)

    def test_it_is_quiet_when_the_two_arms_agree(self):
        self.assertIsNone(check_unity_arm(self._stats(0.1)))

    def test_it_complains_when_they_disagree(self):
        complaint = check_unity_arm(self._stats(0.15))
        self.assertIsNotNone(complaint)
        self.assertIn('WARNING', complaint)

    def test_a_missing_arm_is_not_a_failure(self):
        """--skip-references is a supported way to run the sweep, and it removes the arm the
        check compares against. Nothing to check is not the same as a failed check."""
        stats = self._stats(0.1)
        self.assertIsNone(check_unity_arm(stats[stats['method'] != 'mag_on']))


if __name__ == '__main__':
    unittest.main()
