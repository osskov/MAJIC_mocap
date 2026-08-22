"""`gyro_utils.matrix_rotvec` against scipy, which is the only thing that makes it safe.

It is a hand-rolled replacement for `Rotation.from_matrix(...).as_rotvec()` on the hot path --
worth ~89x there, and ~3x on everything that finite-differences a mocap rotation, which is
most of the pipeline. A fast path that is only *nearly* right about angular velocity would be
a bad trade at any speed, so it is pinned against scipy sample by sample rather than tested
for plausibility.

The tests are in three groups, and the middle one is the one to be careful with:

  THE PHYSICAL BAND. One sample of limb rotation at 40 and 100 Hz, from a slow walk to
  faster than any joint moves. Agreement here is at double precision, so the closed form is
  not an approximation of scipy in this band -- it is the same number.

  THE DEGENERATE BAND. Past theta ~ 84 deg the axis is no longer recoverable from the
  antisymmetric part and `matrix_rotvec` defers to scipy. Real trials DO reach it: 36 samples
  over the 262 IMoVE trials, on foot plates in the long walks. So there is a test that the
  fallback is used and one that the closed form alone would be WRONG without it -- otherwise
  the fallback reads as unreachable and the next person deletes it.

  THE EDGES. Exact identity, denormal angles, short inputs, a duplicated timestamp.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from test.fixtures import require_cache

from src.toolchest.gyro_utils import (_COARSE_ROTATION_COS, finite_difference_rotations,
                                      matrix_rotvec)

# Trials whose rotations the fast path is checked against directly. The long-walk foot plates
# are here on purpose: they are where the held-pose gaps between mocap takes put one-sample
# rotations of up to 159.8 deg, which is the only place in the dataset the fallback fires.
REAL_TRIALS = [('imove', 's3', 't1_walking_001'), ('imove', 's6l', 't12_longwalk_001')]


def scipy_rotvec(matrices):
    """The reference. Deliberately spelled out rather than imported."""
    return Rotation.from_matrix(matrices).as_rotvec()


def rotation_gap(a, b):
    """Angle between two rotvec arrays AS ROTATIONS, in radians.

    Compared this way rather than component-wise because at theta = pi the vectors +v and -v
    describe the same rotation, so a component-wise assertion would fail a correct answer.
    """
    return (Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).magnitude()


class TestAgainstScipy(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def random_axes(self, n):
        axes = self.rng.normal(size=(n, 3))
        return axes / np.linalg.norm(axes, axis=1, keepdims=True)

    def test_matches_scipy_across_the_physical_band(self):
        """One sample of rotation, from a slow walk to faster than any joint turns."""
        for deg_per_s, rate in [(50, 100), (500, 100), (2000, 100), (2000, 40), (3000, 40)]:
            with self.subTest(deg_per_s=deg_per_s, rate=rate):
                theta = np.radians(deg_per_s) / rate
                matrices = Rotation.from_rotvec(self.random_axes(2000) * theta).as_matrix()
                worst = float(np.max(rotation_gap(matrix_rotvec(matrices),
                                                  scipy_rotvec(matrices))))
                self.assertLess(worst, 1e-12,
                                f"{deg_per_s} deg/s at {rate} Hz disagrees with scipy by "
                                f"{worst:.3e} rad")

    def test_matches_scipy_on_uniformly_random_rotations(self):
        """Every branch at once, including the ones physical data never visits."""
        matrices = Rotation.random(5000, rng=1).as_matrix()
        worst = float(np.max(rotation_gap(matrix_rotvec(matrices), scipy_rotvec(matrices))))
        self.assertLess(worst, 1e-12)

    def test_survives_float32_storage(self):
        """The parquet stores rotations as float32, so that is what this is handed."""
        matrices = Rotation.from_rotvec(self.rng.normal(size=(5000, 3)) * 0.01).as_matrix()
        rounded = matrices.astype(np.float32).astype(np.float64)
        worst = float(np.max(rotation_gap(matrix_rotvec(rounded), scipy_rotvec(rounded))))
        # Looser than the tests above, and it has to be: the input itself has only ~1e-7
        # relative precision, so scipy's orthogonalization and this closed form are answering
        # slightly different questions about a matrix that is not quite a rotation.
        self.assertLess(worst, 1e-6)

    def test_identity_and_denormal_angles(self):
        """sin(theta) exactly zero, and small enough to underflow a naive division."""
        self.assertTrue(np.all(matrix_rotvec(np.tile(np.eye(3), (100, 1, 1))) == 0.0))
        for scale in (1e-9, 1e-15, 1e-30):
            with self.subTest(scale=scale):
                matrices = Rotation.from_rotvec(
                    self.rng.normal(size=(500, 3)) * scale).as_matrix()
                got = matrix_rotvec(matrices)
                self.assertTrue(np.all(np.isfinite(got)))
                self.assertLess(float(np.max(rotation_gap(got, scipy_rotvec(matrices)))), 1e-12)


class TestTheFallback(unittest.TestCase):
    """The part a later refactor would be tempted to delete."""

    def setUp(self):
        self.rng = np.random.default_rng(2)
        axes = self.rng.normal(size=(200, 3))
        self.axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)

    def test_coarse_rotations_still_match_scipy(self):
        for deg in [84, 85, 90, 120, 179, 179.99, 180]:
            with self.subTest(deg=deg):
                matrices = Rotation.from_rotvec(self.axes * np.radians(deg)).as_matrix()
                worst = float(np.max(rotation_gap(matrix_rotvec(matrices),
                                                  scipy_rotvec(matrices))))
                self.assertLess(worst, 1e-12,
                                f"theta = {deg} deg disagrees with scipy by {worst:.3e} rad")

    @staticmethod
    def without_fallback(matrices):
        """`matrix_rotvec` with the guard removed, so the failure it prevents is visible."""
        skew = np.stack([matrices[:, 2, 1] - matrices[:, 1, 2],
                         matrices[:, 0, 2] - matrices[:, 2, 0],
                         matrices[:, 1, 0] - matrices[:, 0, 1]], axis=1)
        sin_theta = 0.5 * np.linalg.norm(skew, axis=1)
        cos_theta = 0.5 * (np.trace(matrices, axis1=1, axis2=2) - 1.0)
        theta = np.arctan2(sin_theta, cos_theta)
        return skew * (np.where(sin_theta < 1e-12, 0.5,
                                0.5 * theta / np.maximum(sin_theta, 1e-300)))[:, None]

    def test_the_closed_form_alone_fails_at_pi(self):
        """Why the guard exists, stated as a test rather than as a comment.

        At theta = pi the antisymmetric part is zero, so the closed form takes its
        small-angle branch and returns ZERO for a half turn -- the largest error a rotvec can
        have. The guard is what turns that into scipy's exact answer.
        """
        matrices = Rotation.from_rotvec(self.axes * np.pi).as_matrix()
        self.assertGreater(float(np.max(rotation_gap(self.without_fallback(matrices),
                                                     scipy_rotvec(matrices)))), 3.0)
        self.assertLess(float(np.max(rotation_gap(matrix_rotvec(matrices),
                                                  scipy_rotvec(matrices)))), 1e-12)

    def test_the_closed_form_degrades_gracefully_up_to_pi(self):
        """How the guard's threshold was chosen: it is not where the closed form breaks.

        The error goes as (input precision) / sin(theta), so on the float32 rotations the
        parquet stores it is 1.7e-7 rad at the worst one-sample rotation in the dataset
        (159.8 deg) and only reaches 1e-3 within 0.01 deg of pi. Measured:

            theta       float64 in   float32 in
             90 deg     8.3e-16      3.6e-08
            159.8 deg   1.1e-15      1.7e-07
            179 deg     8.9e-15      3.4e-06
            179.99 deg  8.4e-13      3.9e-04

        So `_COARSE_ROTATION_COS` is deliberately conservative rather than tight: it costs 36
        samples out of 23.9 M and removes the need to reason about how close to pi a corrupt
        marker reconstruction can get. This test pins the shape of that curve, so a future
        threshold change can be argued against numbers.
        """
        for deg, tolerance in [(90, 1e-7), (159.8, 1e-6), (179, 1e-5)]:
            with self.subTest(deg=deg):
                matrices = Rotation.from_rotvec(self.axes * np.radians(deg)).as_matrix()
                rounded = matrices.astype(np.float32).astype(np.float64)
                worst = float(np.max(rotation_gap(self.without_fallback(rounded),
                                                  scipy_rotvec(rounded))))
                self.assertLess(worst, tolerance,
                                f"the closed form is worse at {deg} deg than the curve this "
                                f"test records: {worst:.2e} rad")

    def test_threshold_is_far_outside_what_a_limb_can_do(self):
        """A sanity bound on the constant itself, in the units it will be met in."""
        theta = np.degrees(np.arccos(_COARSE_ROTATION_COS))
        self.assertGreater(theta * 40.0, 3000.0,
                           "the fallback threshold is inside the physical band at 40 Hz, so "
                           "ordinary fast motion would be routed through scipy")


class TestFiniteDifferenceRotations(unittest.TestCase):
    """The caller's own edges, which the rotvec change had to leave alone."""

    def test_short_inputs(self):
        for n in (0, 1, 2, 3):
            with self.subTest(n=n):
                matrices = (Rotation.random(n, rng=3).as_matrix() if n
                            else np.zeros((0, 3, 3)))
                timestamps = np.arange(n) / 100.0
                result = finite_difference_rotations(matrices, timestamps)
                self.assertEqual(result.shape, (max(1, n), 3))
                self.assertTrue(np.all(np.isfinite(result)))

    def test_duplicated_timestamp_does_not_divide_by_zero(self):
        rng = np.random.default_rng(4)
        matrices = Rotation.from_rotvec(rng.normal(size=(50, 3)) * 0.01).as_matrix()
        timestamps = np.arange(50) / 100.0
        timestamps[20] = timestamps[19]
        self.assertTrue(np.all(np.isfinite(finite_difference_rotations(matrices, timestamps))))

    def test_recovers_a_known_angular_velocity(self):
        """End to end: integrate a constant rate, differentiate it back."""
        timestamps = np.arange(500) / 100.0
        omega = np.array([0.3, -1.2, 0.7])
        matrices = Rotation.from_rotvec(np.outer(timestamps, omega)).as_matrix()
        recovered = finite_difference_rotations(matrices, timestamps)
        np.testing.assert_allclose(recovered[:-1], np.tile(omega, (499, 1)), atol=1e-9)


class TestOnRealTrials(unittest.TestCase):
    """Against scipy on the actual mocap rotations, not on synthesized ones.

    Synthetic matrices are exactly orthonormal and smoothly sampled. Real ones are float32,
    come out of a marker fit, and include the held-pose boundaries that produce the only
    coarse rotations in the dataset -- so this is the test that would catch a fast path that
    is fine on textbook input and wrong on ours.
    """

    def plates(self, dataset, session, trial):
        from experiments.experiment_utils import StaleTrialCache, load_trial
        try:
            return load_trial(session, trial, dataset=dataset)
        except StaleTrialCache as stale:
            require_cache(False, str(stale))
        except FileNotFoundError:
            require_cache(False, f"{dataset}/{session}/{trial} is not built")

    def test_matches_scipy_on_built_trials(self):
        for dataset, session, trial in REAL_TRIALS:
            plates = self.plates(dataset, session, trial)
            for name, plate in sorted(plates.items()):
                with self.subTest(trial=f"{session}/{trial}", plate=name):
                    rotations = np.asarray(plate.world_trace.rotations, dtype=np.float64)
                    relative = np.matmul(rotations[:-1].transpose(0, 2, 1), rotations[1:])
                    worst = float(np.max(rotation_gap(matrix_rotvec(relative),
                                                      scipy_rotvec(relative))))
                    # Set off the measured worst case over all 3541 plates of the dataset,
                    # 1.3e-8 rad, which is where float32 storage puts the floor -- not off
                    # double precision, which only the synthetic tests above can reach. In the
                    # units this ends up in: 1e-7 rad between consecutive 100 Hz samples is
                    # 1e-5 deg/s against gyro signals of 50 to 9800 deg/s.
                    self.assertLess(worst, 1e-7,
                                    f"{session}/{trial}/{name} disagrees with scipy by "
                                    f"{worst:.3e} rad")

    def test_the_long_walks_actually_reach_the_fallback(self):
        """The finding that makes the fallback load-bearing, pinned so it stays visible.

        If this ever fails because no sample is coarse any more, the fallback has become
        genuinely unreachable and can go -- but that is a claim about the DATA, and it should
        have to be re-established rather than assumed.
        """
        plates = self.plates('imove', 's6l', 't12_longwalk_001')
        coarse = 0
        for plate in plates.values():
            rotations = np.asarray(plate.world_trace.rotations, dtype=np.float64)
            relative = np.matmul(rotations[:-1].transpose(0, 2, 1), rotations[1:])
            cosines = 0.5 * (np.trace(relative, axis1=1, axis2=2) - 1.0)
            coarse += int((cosines < _COARSE_ROTATION_COS).sum())
        self.assertGreater(coarse, 0,
                           "no one-sample rotation in s6l/t12_longwalk_001 is past the "
                           "fallback threshold, so this trial no longer exercises it")


if __name__ == '__main__':
    unittest.main()
