"""The bone-to-cluster comparison, and the confound that decides what it means.

Driven with synthetic poses where the answer is known, because the whole point of the module
is separating two mechanisms that look identical in the summary statistics -- soft-tissue
artifact and residual time misalignment both grow with angular speed. A test that only checked
"the artifact is nonzero" could not tell the module from one that measures nothing but sync.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.bone_vs_cluster import (_constant_rotation, artifact_for_site,
                                         MIN_PAIRED_FRAMES)


class _Trace:
    def __init__(self, rotations, positions, valid):
        self.rotations, self.positions, self.valid = rotations, positions, valid


class _Plate:
    """The two attributes artifact_for_site reads, without building a real PlateTrial."""
    def __init__(self, rotations, gyro, valid):
        self.valid = valid
        self.world_trace = _Trace(rotations, np.zeros((len(rotations), 3)), valid)
        self.imu_trace = type('imu', (), {'gyro': gyro})()


def _motion(n=400, rate=150.0, amplitude=0.6, freq=1.1):
    """A rotating segment and the angular velocity that goes with it."""
    time = np.arange(n) / rate
    angle = amplitude * np.sin(2 * np.pi * freq * time)
    rotations = Rotation.from_rotvec(np.outer(angle, [0.0, 0.0, 1.0])).as_matrix()
    rate_of_change = (amplitude * 2 * np.pi * freq
                      * np.cos(2 * np.pi * freq * time))
    gyro = np.outer(rate_of_change, [0.0, 0.0, 1.0])
    return time, rotations, gyro


class TestConstantFrameRelationIsRemoved(unittest.TestCase):
    """The bone frame and the cluster frame are differently defined, and most of the raw
    difference between them is that definition rather than any motion."""

    def test_a_pure_frame_offset_reports_no_artifact(self):
        _, rotations, gyro = _motion()
        offset = Rotation.from_euler('xyz', [25.0, -40.0, 60.0], degrees=True).as_matrix()
        valid = np.ones(len(rotations), dtype=bool)
        out = artifact_for_site(_Plate(rotations, gyro, valid),
                                _Plate(rotations @ offset, gyro, valid))
        self.assertLess(out['rotation_artifact_median_deg'], 1e-6)

    def test_the_recovered_constant_is_the_one_applied(self):
        _, rotations, _ = _motion()
        offset = Rotation.from_euler('xyz', [10.0, 20.0, -30.0], degrees=True).as_matrix()
        np.testing.assert_allclose(_constant_rotation(rotations, rotations @ offset),
                                   offset, atol=1e-9)


class TestTheConfoundIsSeparated(unittest.TestCase):
    """Soft tissue and a time offset both scale with speed. Only the AXIS tells them apart."""

    def test_a_time_offset_is_reported_as_aligned_with_omega(self):
        """Shifting one trace in time rotates it about the instantaneous angular-velocity
        vector, so |cos| goes to 1. This is the signature the module must recognise, because
        the size of the artifact alone cannot."""
        _, rotations, gyro = _motion()
        shift = 6
        valid = np.ones(len(rotations) - shift, dtype=bool)
        out = artifact_for_site(_Plate(rotations[:-shift], gyro[:-shift], valid),
                                _Plate(rotations[shift:], gyro[:-shift], valid))
        self.assertGreater(out['axis_alignment_with_omega'], 0.95)

    def test_the_equivalent_offset_recovers_the_shift_applied(self):
        rate, shift = 150.0, 6
        _, rotations, gyro = _motion(rate=rate)
        valid = np.ones(len(rotations) - shift, dtype=bool)
        out = artifact_for_site(_Plate(rotations[:-shift], gyro[:-shift], valid),
                                _Plate(rotations[shift:], gyro[:-shift], valid))
        self.assertAlmostEqual(out['equivalent_offset_ms'], 1000 * shift / rate, delta=8.0)

    def test_an_artifact_off_the_omega_axis_is_not_called_timing(self):
        """Soft tissue displaces where the anatomy lets it, not along omega. Here the
        perturbation is deliberately perpendicular to the rotation axis, and the module must
        not attribute it to sync."""
        _, rotations, gyro = _motion()
        n = len(rotations)
        wobble = Rotation.from_rotvec(
            np.outer(0.06 * np.sin(2 * np.pi * 3.0 * np.arange(n) / 150.0),
                     [1.0, 0.0, 0.0])).as_matrix()
        valid = np.ones(n, dtype=bool)
        out = artifact_for_site(_Plate(rotations, gyro, valid),
                                _Plate(rotations @ wobble, gyro, valid))
        self.assertGreater(out['rotation_artifact_median_deg'], 0.5)
        self.assertLess(out['axis_alignment_with_omega'], 0.4)


class TestGuards(unittest.TestCase):
    def test_too_little_overlap_returns_nothing(self):
        """A short overlap cannot separate the constant frame relation from the motion around
        it, so the 'artifact' would be mostly fitting error."""
        _, rotations, gyro = _motion(n=MIN_PAIRED_FRAMES - 1)
        valid = np.ones(len(rotations), dtype=bool)
        self.assertIsNone(artifact_for_site(_Plate(rotations, gyro, valid),
                                            _Plate(rotations, gyro, valid)))

    def test_only_frames_valid_in_both_are_scored(self):
        """The biplane window is a second inside a ten-second Vicon capture. Scoring a frame
        where only one reference is valid would compare a pose against an extrapolation."""
        _, rotations, gyro = _motion()
        both = np.ones(len(rotations), dtype=bool)
        half = both.copy(); half[len(rotations) // 2:] = False
        full = artifact_for_site(_Plate(rotations, gyro, both),
                                 _Plate(rotations, gyro, both))
        part = artifact_for_site(_Plate(rotations, gyro, both),
                                 _Plate(rotations, gyro, half))
        self.assertEqual(full['n_paired_frames'], len(rotations))
        self.assertEqual(part['n_paired_frames'], int(half.sum()))


if __name__ == '__main__':
    unittest.main()
