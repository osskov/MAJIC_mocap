import json
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

os.environ.setdefault("DISABLE_TQDM", "True")

import paths
from experiments import experiment_utils as eu
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest import trial_io
from src.toolchest.building import alborno, sources


def make_plates(names=('femur_r_imu', 'tibia_r_imu'), num_samples=200, seed=0, valid=None):
    """A trial's worth of plates with non-trivial, non-degenerate content.

    Real rotations (not identity) and distinct per-plate data, so a serializer that
    dropped a channel, transposed a matrix or crossed two plates would fail rather
    than round-trip zeros successfully.
    """
    rng = np.random.default_rng(seed)
    timestamps = np.arange(num_samples) / 100.0
    plates = {}
    for i, name in enumerate(names):
        angles = np.cumsum(rng.normal(0, 0.02, size=(num_samples, 3)), axis=0) + i
        rotations = Rotation.from_euler('xyz', angles).as_matrix()
        world = WorldTrace(timestamps, rng.normal(0, 1, (num_samples, 3)), rotations,
                           valid=None if valid is None else valid[i])
        imu = IMUTrace(timestamps,
                       gyro=rng.normal(0, 1, (num_samples, 3)),
                       acc=rng.normal(0, 9.81, (num_samples, 3)),
                       mag=rng.normal(0, 1, (num_samples, 3)))
        plates[name] = PlateTrial(name, imu, world)
    return plates


class TestFrameRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_every_channel(self):
        plates = make_plates()
        back = trial_io.plates_from_frame(trial_io.plates_to_frame(plates))

        self.assertEqual(set(back), set(plates))
        for name, original in plates.items():
            restored = back[name]
            self.assertEqual(len(restored), len(original))
            # float32 storage: ~1e-7 relative. Absolute tolerances are set off the
            # channel magnitudes (acc runs to ~30 m/s^2, rotations to 1).
            np.testing.assert_allclose(restored.imu_trace.acc, original.imu_trace.acc, atol=1e-5)
            np.testing.assert_allclose(restored.imu_trace.gyro, original.imu_trace.gyro, atol=1e-6)
            np.testing.assert_allclose(restored.imu_trace.mag, original.imu_trace.mag, atol=1e-6)
            np.testing.assert_allclose(restored.world_trace.positions, original.world_trace.positions, atol=1e-6)
            np.testing.assert_allclose(restored.world_trace.rotations, original.world_trace.rotations, atol=1e-6)

    def test_timestamps_are_exact(self):
        """float64 timestamps, not float32 — PlateTrial asserts agreement to 1e-8."""
        plates = make_plates(num_samples=2000)
        back = trial_io.plates_from_frame(trial_io.plates_to_frame(plates))
        for name, original in plates.items():
            np.testing.assert_array_equal(back[name].imu_trace.timestamps,
                                          original.imu_trace.timestamps)
            # The two traces share one column, so the constructor's assert is exact.
            self.assertEqual(
                np.abs(back[name].imu_trace.timestamps - back[name].world_trace.timestamps).max(), 0.0)

    def test_rotations_stay_orthonormal(self):
        plates = make_plates()
        back = trial_io.plates_from_frame(trial_io.plates_to_frame(plates))
        for restored in back.values():
            R = restored.world_trace.rotations
            deviation = np.abs(np.einsum('nij,nkj->nik', R, R) - np.eye(3)).max()
            self.assertLess(deviation, 1e-5)

    def test_plates_are_not_crossed(self):
        """Distinct plates must come back distinct — catches a bad groupby or reshape."""
        plates = make_plates(names=('a', 'b', 'c'))
        back = trial_io.plates_from_frame(trial_io.plates_to_frame(plates))
        self.assertFalse(np.allclose(back['a'].imu_trace.acc, back['b'].imu_trace.acc))
        np.testing.assert_allclose(back['c'].world_trace.rotations,
                                   plates['c'].world_trace.rotations, atol=1e-6)

    def test_rotation_columns_are_row_major(self):
        """rot_ij is row i, column j — a transpose here would silently invert every frame."""
        plates = make_plates(names=('only',), num_samples=10)
        frame = trial_io.plates_to_frame(plates)
        expected = plates['only'].world_trace.rotations
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(frame[f'rot_{i}{j}'].to_numpy(), expected[:, i, j], atol=1e-6)

    def test_validity_mask_round_trips_per_plate(self):
        """Distinct masks per plate — a shared or broadcast mask would pass a weaker test."""
        a_valid, b_valid = np.ones(200, dtype=bool), np.ones(200, dtype=bool)
        a_valid[10:20] = False
        b_valid[150:155] = False
        plates = make_plates(names=('a', 'b'), valid=[a_valid, b_valid])

        back = trial_io.plates_from_frame(trial_io.plates_to_frame(plates))

        np.testing.assert_array_equal(back['a'].valid, a_valid)
        np.testing.assert_array_equal(back['b'].valid, b_valid)

    def test_all_valid_round_trips(self):
        back = trial_io.plates_from_frame(trial_io.plates_to_frame(make_plates()))
        for plate in back.values():
            self.assertTrue(plate.valid.all())

    def test_valid_column_is_bool_not_float(self):
        """A float column would survive round-tripping but compare wrong under `&`."""
        frame = trial_io.plates_to_frame(make_plates())
        self.assertEqual(frame['valid'].dtype, np.dtype(bool))

    def test_a_v1_artifact_without_the_mask_is_rejected(self):
        """Silently defaulting a maskless artifact to all-True would assert that
        interpolated and corrupt frames are measured ground truth."""
        frame = trial_io.plates_to_frame(make_plates()).drop(columns=['valid'])
        with self.assertRaises(ValueError):
            trial_io.plates_from_frame(frame)

    def test_empty_trial_raises(self):
        with self.assertRaises(ValueError):
            trial_io.plates_to_frame({})

    def test_missing_column_raises(self):
        frame = trial_io.plates_to_frame(make_plates()).drop(columns=['mag_z'])
        with self.assertRaises(ValueError):
            trial_io.plates_from_frame(frame)


class TestCacheKey(unittest.TestCase):
    """The key exists to make staleness a cache MISS rather than a wrong answer."""

    def setUp(self):
        self.folder = paths.raw_trial_dir('01', 'walking')
        if not self.folder.is_dir():
            self.skipTest(f"source data not present at {self.folder}")

    def test_key_is_stable_across_calls(self):
        self.assertEqual(eu.trial_cache_key('01', 'walking', align=True),
                         eu.trial_cache_key('01', 'walking', align=True))

    def test_align_flag_changes_the_key(self):
        """align_plate_trials rewrites every rotation, so it cannot share an entry."""
        self.assertNotEqual(eu.trial_cache_key('01', 'walking', align=True),
                            eu.trial_cache_key('01', 'walking', align=False))

    def test_inventory_covers_what_the_loader_reads_and_no_more(self):
        names = {s['name'] for s in eu.trial_cache_key('01', 'walking', align=True)['sources']}
        self.assertIn('walking.trc', names)
        self.assertIn('imu data/femur_r_imu.txt', names)
        # The .mtb is never parsed and the madgwick outputs have colliding filenames;
        # including either would invalidate the cache when an unread file moved.
        self.assertFalse([n for n in names if n.endswith('.mtb')], names)
        self.assertFalse([n for n in names if 'madgwick' in n], names)

    def test_toolchest_edit_invalidates(self):
        """A change to the code that BUILDS a trial must not reuse an old artifact."""
        with mock.patch.object(eu, '_CONTENT_MODULES', ('PlateTrial.py',)):
            eu._toolchest_digest.cache_clear()
            narrowed = eu._toolchest_digest()
        eu._toolchest_digest.cache_clear()
        self.assertNotEqual(narrowed, eu._toolchest_digest())


class TestCacheStatus(unittest.TestCase):
    """Status reporting, against a throwaway dataset registered for the test.

    A registered TrialSource is now required — the cache layer asks it where the trial's
    files are and which of them the key should hash, instead of templating a path. So a
    test dataset has to be a real entry in the registry rather than just a directory name.
    """

    DATASET = '_test_dataset'

    def setUp(self):
        if not paths.raw_trial_dir('01', 'walking').is_dir():
            self.skipTest("source data not present")
        real = sources.get_source('alborno')
        self.source = sources.TrialSource(
            name=self.DATASET,
            enumerate_trials=lambda: [('01', 'walking')],
            source_dir=real.source_dir,          # points at the real Al Borno trial
            source_globs=real.source_globs,
            load=real.load,
        )
        self._patch = mock.patch.dict(sources.SOURCES, {self.DATASET: self.source})
        self._patch.start()
        self.tmp = paths.TRIALS_DIR / self.DATASET

    def tearDown(self):
        import shutil
        self._patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _status(self):
        return eu.cached_trial_status('01', 'walking', dataset=self.DATASET)

    def _path(self):
        return paths.cached_trial_path(self.DATASET, '01', 'walking')

    def test_missing_then_fresh(self):
        self.assertEqual(self._status()[0], 'missing')
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        self.assertEqual(self._status()[0], 'fresh')

    def test_key_mismatch_reports_stale_and_names_the_field(self):
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        manifest_path = paths.manifest_path(self._path())
        manifest = json.loads(manifest_path.read_text())
        manifest['cache_key']['toolchest_digest'] = 'deadbeefdeadbeef'
        manifest_path.write_text(json.dumps(manifest))

        status, reason = self._status()
        self.assertEqual(status, 'stale')
        self.assertIn('toolchest_digest', reason)

    def test_a_stale_entry_raises_rather_than_loading(self):
        """The whole point of the strict loader: staleness is loud, not a silent reparse."""
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        manifest_path = paths.manifest_path(self._path())
        manifest = json.loads(manifest_path.read_text())
        manifest['cache_key']['schema_version'] = -1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(eu.StaleTrialCache):
            eu.load_trial('01', 'walking', dataset=self.DATASET)

    def test_missing_manifest_is_stale_not_fresh(self):
        """A parquet with no sidecar has unknown provenance; refuse to trust it."""
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        paths.manifest_path(self._path()).unlink()
        self.assertEqual(self._status(), ('stale', 'no manifest sidecar'))

    def test_save_records_diagnostics(self):
        path = eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        diagnostics = paths.read_manifest(path)['diagnostics']
        self.assertEqual(diagnostics['n_plates'], 2)
        self.assertAlmostEqual(diagnostics['sample_rate_hz'], 100.0, places=6)
        for stats in diagnostics['plates'].values():
            self.assertIn('gyro_residual_lowpass_rms_deg_s', stats)


class TestStrictLoading(unittest.TestCase):
    """load_trial reads the artifact and nothing else.

    These tests used to assert the opposite — that a cache miss silently fell back to
    parsing from source. That made a stale artifact cost time and nothing else, which also
    made it invisible: a sweep could mix cached and freshly-parsed trials with no record of
    which produced which number.
    """

    def test_a_missing_trial_raises_and_names_the_build_command(self):
        with mock.patch.object(eu, 'cached_trial_status', return_value=('missing', None)):
            with self.assertRaises(eu.StaleTrialCache) as ctx:
                eu.load_trial('01', 'walking')
        self.assertIn('build_trials', str(ctx.exception))
        self.assertIn('--dataset alborno', str(ctx.exception))

    def test_a_stale_trial_raises_with_the_reason(self):
        with mock.patch.object(eu, 'cached_trial_status',
                               return_value=('stale', 'toolchest_digest: ...')):
            with self.assertRaises(eu.StaleTrialCache) as ctx:
                eu.load_trial('01', 'walking')
        self.assertIn('toolchest_digest', str(ctx.exception))

    def test_it_never_touches_the_source_reader(self):
        """No fallback path exists, so a fresh read must not reach the dataset reader."""
        if not paths.raw_trial_dir('01', 'walking').is_dir():
            self.skipTest("source data not present")
        if eu.cached_trial_status('01', 'walking')[0] != 'fresh':
            self.skipTest("trial cache not built; run python -m experiments.build_trials")
        with mock.patch.object(sources.alborno, 'load_trial') as reader:
            plates = eu.load_trial('01', 'walking')
        reader.assert_not_called()
        self.assertGreater(len(plates), 0)

    def test_load_raw_data_still_works_as_a_deprecated_alias(self):
        with mock.patch.object(eu, 'load_trial', return_value={'x': None}) as inner:
            eu.load_raw_data('01', 'walking')
        inner.assert_called_once()


class TestRealTrialRoundTrip(unittest.TestCase):
    """End-to-end against real source data, if it is present."""

    def _live(self, subject, trial):
        """Straight from the dataset reader — what build_trials does."""
        if not paths.raw_trial_dir(subject, trial).is_dir():
            self.skipTest("source data not present")
        return sources.get_source('alborno').load(subject, trial, True)

    def test_cached_trial_matches_a_live_load(self):
        live = self._live('01', 'walking')
        back = trial_io.plates_from_frame(trial_io.plates_to_frame(live))

        self.assertEqual(set(back), set(live))
        for name, original in live.items():
            np.testing.assert_allclose(back[name].imu_trace.acc, original.imu_trace.acc, atol=1e-4)
            np.testing.assert_allclose(back[name].world_trace.rotations,
                                       original.world_trace.rotations, atol=1e-6)
            np.testing.assert_array_equal(back[name].valid, original.valid)

    def test_a_known_damaged_trial_round_trips_its_mask(self):
        """Subject08/walking/calcn_l is masked; the artifact must not launder that away."""
        live = self._live('08', 'walking')
        self.assertFalse(live['calcn_l_imu'].valid.all(),
                         "fixture assumption broken: this plate should carry known damage")

        back = trial_io.plates_from_frame(trial_io.plates_to_frame(live))
        np.testing.assert_array_equal(back['calcn_l_imu'].valid, live['calcn_l_imu'].valid)
        # Every plate now carries a mask, because alignment keeps the inertial record from
        # before and after the mocap window and marks it unscoreable — so "clean" no longer
        # means all-True. What must hold is that the mask survives the round trip unchanged.
        np.testing.assert_array_equal(back['femur_r_imu'].valid, live['femur_r_imu'].valid)
        self.assertFalse(back['femur_r_imu'].valid.all(),
                         "fixture assumption: this trial has inertial data outside the "
                         "mocap window, which is retained and marked unscoreable")

    def test_masked_frames_survive_sync_and_alignment(self):
        """assemble_plate_trials pads and align_world_to_imu rebuilds the WorldTrace;
        either dropping the mask would be silent."""
        plate = self._live('08', 'walking')['calcn_l_imu']
        self.assertEqual(len(plate.valid), len(plate))
        self.assertFalse(plate.valid.all())
