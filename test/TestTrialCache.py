import json
import os
import unittest
import warnings

from test.fixtures import require_cache, require_data
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
            require_data(False, f"no source data at {self.folder}")

    def test_key_is_stable_across_calls(self):
        self.assertEqual(eu.trial_cache_key('01', 'walking'),
                         eu.trial_cache_key('01', 'walking'))

    def test_content_size_is_part_of_the_key(self):
        """A truncated parquet has to be a MISS, not a wrong answer.

        cached_trial_status never opens the artifact -- it compares manifest fields. Without
        the row and plate counts in the key, a write interrupted between the parquet and the
        manifest leaves a short file validating against the old manifest, fresh forever.
        """
        full = eu.trial_cache_key('01', 'walking', content={'n_rows': 1000, 'n_plates': 8})
        short = eu.trial_cache_key('01', 'walking', content={'n_rows': 500, 'n_plates': 8})
        missing = eu.trial_cache_key('01', 'walking', content={'n_rows': 1000, 'n_plates': 7})

        self.assertNotEqual(full, short)
        self.assertNotEqual(full, missing)

    def test_inventory_covers_what_the_loader_reads_and_no_more(self):
        names = {s['name'] for s in eu.trial_cache_key('01', 'walking')['sources']}
        self.assertIn('walking.trc', names)
        self.assertIn('imu data/femur_r_imu.txt', names)
        # The .mtb is never parsed and the madgwick outputs have colliding filenames;
        # including either would invalidate the cache when an unread file moved.
        self.assertFalse([n for n in names if n.endswith('.mtb')], names)
        self.assertFalse([n for n in names if 'madgwick' in n], names)

    def test_toolchest_edit_invalidates(self):
        """A change to the code that BUILDS a trial must not reuse an old artifact."""
        with mock.patch.object(eu, '_CORE_MODULES', ('PlateTrial.py',)):
            eu._toolchest_digest.cache_clear()
            narrowed = eu._toolchest_digest('alborno')
        eu._toolchest_digest.cache_clear()
        self.assertNotEqual(narrowed, eu._toolchest_digest('alborno'))

    def test_a_comment_edit_does_not_invalidate(self):
        """The digest hashes the AST, not the bytes.

        Raw-byte hashing made every prose edit a full rebuild of both datasets, and this
        codebase is deliberately comment-dense -- most invalidations were re-explaining
        something rather than changing behaviour.
        """
        module = paths.REPO_ROOT / 'src' / 'toolchest' / 'building' / 'assembly.py'
        original = module.read_bytes()
        eu._toolchest_digest.cache_clear()
        before = eu._toolchest_digest('alborno')
        try:
            module.write_bytes(original + b"\n# explanatory comment added later\n")
            eu._toolchest_digest.cache_clear()
            self.assertEqual(eu._toolchest_digest('alborno'), before)
        finally:
            module.write_bytes(original)
            eu._toolchest_digest.cache_clear()

    def test_a_semantic_edit_still_invalidates(self):
        """The property that must survive the above. Comment-insensitivity is only safe if
        behaviour-sensitivity is intact."""
        module = paths.REPO_ROOT / 'src' / 'toolchest' / 'building' / 'assembly.py'
        original = module.read_bytes()
        eu._toolchest_digest.cache_clear()
        before = eu._toolchest_digest('alborno')
        try:
            module.write_bytes(original.replace(b'SYNC_SPREAD_LIMIT_S = 1.0',
                                                b'SYNC_SPREAD_LIMIT_S = 2.0'))
            eu._toolchest_digest.cache_clear()
            self.assertNotEqual(eu._toolchest_digest('alborno'), before)
        finally:
            module.write_bytes(original)
            eu._toolchest_digest.cache_clear()

    def test_one_dataset_s_reader_does_not_invalidate_the_other(self):
        """alborno.py was invalidating all 262 IMoVE artifacts. Fail-closed, but needlessly:
        the IMoVE loader never reads it."""
        module = paths.REPO_ROOT / 'src' / 'toolchest' / 'building' / 'alborno.py'
        original = module.read_bytes()
        eu._toolchest_digest.cache_clear()
        before_alborno = eu._toolchest_digest('alborno')
        before_imove = eu._toolchest_digest('imove')
        try:
            module.write_bytes(original.replace(b'MARKER_FAULT_THRESHOLD = 2.0',
                                                b'MARKER_FAULT_THRESHOLD = 3.0'))
            eu._toolchest_digest.cache_clear()
            self.assertNotEqual(eu._toolchest_digest('alborno'), before_alborno)
            self.assertEqual(eu._toolchest_digest('imove'), before_imove)
        finally:
            module.write_bytes(original)
            eu._toolchest_digest.cache_clear()

    def test_the_digest_does_not_move_with_the_interpreter(self):
        """ast.dump emits internal field names that change between CPython releases, so 3.12
        and 3.13 produced different digests for identical source -- two interpreters on one
        machine each invalidated the other's 281 artifacts. ast.unparse emits canonical
        source instead. Pinned by round-tripping the module through it: a digest built on
        anything version-dependent would not survive this.
        """
        import ast
        module = paths.REPO_ROOT / 'src' / 'toolchest' / 'building' / 'assembly.py'
        once = eu._semantic_source(module)
        # Re-parsing unparsed source must give the same text back, which is the property
        # that makes it stable rather than merely different from dump.
        twice = ast.unparse(ast.parse(once)).encode()
        self.assertEqual(once.decode().strip(), twice.decode().strip())

    def test_every_reader_module_is_hashed_by_someone(self):
        """A reader missing from _READER_MODULES is a silent stale cache for its dataset."""
        for name in sources.SOURCES:
            self.assertIn(name, eu._READER_MODULES,
                          f"{name} has no entry, so its reader is unhashed")


class TestCacheStatus(unittest.TestCase):
    """Status reporting, against a throwaway dataset registered for the test.

    A registered TrialSource is now required — the cache layer asks it where the trial's
    files are and which of them the key should hash, instead of templating a path. So a
    test dataset has to be a real entry in the registry rather than just a directory name.
    """

    DATASET = '_test_dataset'

    def setUp(self):
        if not paths.raw_trial_dir('01', 'walking').is_dir():
            require_data(False, "no source data")
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

    def test_a_report_is_written_without_a_parquet_beside_it(self):
        """The failure path calls save_build_report on its own, with no artifact to sit
        beside, so it has to create its own parent directory and be readable afterwards."""
        import pandas as pd
        from src.toolchest.building.report import BuildReport

        report = BuildReport()
        report.add('S1_parse', 'file', 'thigh_r.txt', missing_samples=207)
        path = eu.save_build_report(report, '01', 'walking', dataset=self.DATASET)

        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        self.assertFalse(self._path().exists(), 'no parquet should have been created')
        self.assertEqual(list(pd.read_parquet(path).metric), ['missing_samples'])

    def test_an_empty_report_writes_no_sidecar(self):
        """An empty file would read as 'instrumented, and found nothing' rather than 'this
        build raised before measuring anything', which is the opposite conclusion."""
        from src.toolchest.building.report import BuildReport
        self.assertIsNone(eu.save_build_report(BuildReport(), '01', 'walking',
                                               dataset=self.DATASET))
        self.assertFalse(eu.build_report_path(self.DATASET, '01', 'walking').exists())

    def test_a_truncated_parquet_is_stale_not_fresh(self):
        """The guard where the work happens, not one level above it.

        cached_trial_status compares manifest fields and never opened the artifact, so a
        write interrupted between the parquet and the manifest left a short file validating
        against the OLD manifest -- and because --force means the cache key is unchanged, it
        validated forever. Truncating in place with the manifest untouched reproduces exactly
        that state.
        """
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        self.assertEqual(self._status(), ('fresh', None))

        path = self._path()
        with open(path, 'r+b') as handle:
            handle.truncate(path.stat().st_size // 2)

        status, reason = self._status()
        self.assertEqual(status, 'stale')
        self.assertIn('n_rows', reason)

    def test_a_manifest_without_the_content_check_is_stale(self):
        """Artifacts written before the check existed cannot be trusted, because the very
        thing they lack is the evidence that they are whole."""
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        manifest_path = paths.manifest_path(self._path())
        manifest = json.loads(manifest_path.read_text())
        manifest['cache_key'].pop('n_rows')
        manifest_path.write_text(json.dumps(manifest))

        self.assertEqual(self._status(), ('stale', 'manifest predates the content check'))

    def test_row_count_of_an_unreadable_file_is_none_rather_than_raising(self):
        """A corrupt parquet has to report a miss, not blow up inside a status check that
        the build loop runs over every trial."""
        path = self._path()
        paths.ensure_parent(path).write_bytes(b'this is not a parquet')
        self.assertIsNone(eu._parquet_row_count(path))

    def test_the_staging_file_does_not_survive_a_successful_write(self):
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        staging = self._path().with_suffix(self._path().suffix + '.tmp')
        self.assertFalse(staging.exists(), f"{staging.name} was left behind")

    def test_a_failed_write_leaves_the_previous_artifact_readable(self):
        """The point of writing to a temporary name and renaming. Before, a failure partway
        through to_parquet left a half-file where the good one had been."""
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        good_bytes = self._path().read_bytes()

        with mock.patch.object(eu.trial_io, 'plates_to_frame',
                               side_effect=RuntimeError('disk full')):
            with self.assertRaises(RuntimeError):
                eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)

        self.assertEqual(self._path().read_bytes(), good_bytes)
        self.assertEqual(self._status(), ('fresh', None))

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

    @staticmethod
    def _consistent_plates(twist_deg=0.0):
        """Plates whose IMU genuinely derives from their world trace.

        make_plates() pairs unrelated traces -- its raw residual is already 222 deg/s -- so a
        deliberate misalignment cannot be seen against it. The residual only means anything
        when the aligned case is near zero.
        """
        timestamps = np.arange(400) / 100.0
        angles = np.column_stack([0.9 * np.sin(2 * np.pi * 0.7 * timestamps),
                                  0.6 * np.sin(2 * np.pi * 1.1 * timestamps),
                                  0.4 * np.sin(2 * np.pi * 1.9 * timestamps)])
        rotations = Rotation.from_euler('zyx', angles).as_matrix()
        world = WorldTrace(timestamps, np.zeros((len(timestamps), 3)), rotations)
        synthetic = world.calculate_imu_trace(skip_lin_acc=True)
        imu = IMUTrace(timestamps, synthetic.gyro, synthetic.acc, synthetic.mag)

        twist = Rotation.from_euler('y', twist_deg, degrees=True).as_matrix()
        twisted = WorldTrace(timestamps, world.positions,
                             np.matmul(rotations, twist), valid=world.valid)
        return {'femur_r_imu': PlateTrial('femur_r_imu', imu, twisted)}

    def test_the_alignment_residual_responds_to_misalignment(self):
        """The diagnostic is checked for PRESENCE everywhere and for MEANING nowhere.

        It is the triage number for 281 artifacts and the thing RESIDUAL_WARN_DEG_S gates on,
        so a change that left it always near zero would silence the warning across the whole
        tree without failing anything.
        """
        good = eu.trial_diagnostics(self._consistent_plates(0.0))['plates']
        bad = eu.trial_diagnostics(self._consistent_plates(20.0))['plates']

        for name in good:
            self.assertLess(good[name]['gyro_residual_raw_rms_deg_s'], 1.0,
                            "an aligned plate must sit near zero or the number says nothing")
            self.assertGreater(bad[name]['gyro_residual_raw_rms_deg_s'],
                               10 * good[name]['gyro_residual_raw_rms_deg_s'],
                               f"{name}: 20 deg of misalignment did not move the residual")

    def test_suspect_lists_the_plates_over_the_threshold(self):
        """The field load_trial warns on. Derived at write time, so if it stopped being
        derived the warning would simply never fire."""
        clean = eu.trial_diagnostics(self._consistent_plates(0.0))
        self.assertIn('suspect', clean)
        self.assertEqual(clean['suspect'], [])

        self.assertEqual(eu.trial_diagnostics(self._consistent_plates(60.0))['suspect'],
                         ['femur_r_imu'])

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
        with mock.patch.object(eu, '_cached_trial_state',
                               return_value=('missing', None, None)):
            with self.assertRaises(eu.StaleTrialCache) as ctx:
                eu.load_trial('01', 'walking')
        self.assertIn('build_trials', str(ctx.exception))
        self.assertIn('--dataset alborno', str(ctx.exception))

    def test_a_stale_trial_raises_with_the_reason(self):
        with mock.patch.object(eu, '_cached_trial_state',
                               return_value=('stale', 'toolchest_digest: ...', None)):
            with self.assertRaises(eu.StaleTrialCache) as ctx:
                eu.load_trial('01', 'walking')
        self.assertIn('toolchest_digest', str(ctx.exception))

    def test_it_never_touches_the_source_reader(self):
        """No fallback path exists, so a fresh read must not reach the dataset reader."""
        if not paths.raw_trial_dir('01', 'walking').is_dir():
            require_data(False, "no source data")
        if eu.cached_trial_status('01', 'walking')[0] != 'fresh':
            require_cache(False, "trial cache not built or stale")
        with mock.patch.object(sources.alborno, 'load_trial') as reader:
            plates = eu.load_trial('01', 'walking')
        reader.assert_not_called()
        self.assertGreater(len(plates), 0)

    def test_load_raw_data_still_works_as_a_deprecated_alias(self):
        with mock.patch.object(eu, 'load_trial', return_value={'x': None}) as inner:
            eu.load_raw_data('01', 'walking')
        inner.assert_called_once()


class TestSuspectReporting(unittest.TestCase):
    """The warning is the whole point of the `suspect` field, and had no tests.

    trial_diagnostics' producer was covered; the consumer was not. The purpose of this API is
    that an alignment failure reaches the person computing joint angles rather than dying in
    a build log an hour earlier -- so "does it actually reach them" is the assertion that
    matters. The manifest is JSON, so a suspect trial can be staged by editing it.
    """

    DATASET = '_suspect_dataset'

    def setUp(self):
        real = sources.get_source('alborno')
        self.source = sources.TrialSource(
            name=self.DATASET, enumerate_trials=lambda: [('01', 'walking')],
            source_dir=real.source_dir, source_globs=real.source_globs, load=real.load)
        self._patch = mock.patch.dict(sources.SOURCES, {self.DATASET: self.source})
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.tmp = paths.TRIALS_DIR / self.DATASET
        self.addCleanup(lambda: __import__('shutil').rmtree(self.tmp, ignore_errors=True))

        require_data(paths.raw_trial_dir('01', 'walking').is_dir(), "no source data")
        eu.save_cached_trial(make_plates(), '01', 'walking', dataset=self.DATASET)
        self.path = paths.cached_trial_path(self.DATASET, '01', 'walking')

    def _set_suspect(self, names):
        manifest_path = paths.manifest_path(self.path)
        manifest = json.loads(manifest_path.read_text())
        manifest['diagnostics']['suspect'] = names
        manifest_path.write_text(json.dumps(manifest))

    def test_a_clean_trial_neither_warns_nor_raises(self):
        self._set_suspect([])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            eu.load_trial('01', 'walking', dataset=self.DATASET)
        self.assertEqual([w for w in caught
                          if issubclass(w.category, eu.SuspectTrialWarning)], [])

    def test_a_suspect_trial_warns_and_names_the_plates(self):
        self._set_suspect(['femur_r_imu', 'tibia_r_imu'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            eu.load_trial('01', 'walking', dataset=self.DATASET)

        suspect = [w for w in caught if issubclass(w.category, eu.SuspectTrialWarning)]
        self.assertEqual(len(suspect), 1)
        self.assertIn('femur_r_imu', str(suspect[0].message))
        self.assertIn('tibia_r_imu', str(suspect[0].message))

    def test_strict_raises_instead_of_warning(self):
        """For callers that would rather compute no joint angle than one from a plate whose
        mocap and IMU disagree by twice the worst normal amount."""
        self._set_suspect(['femur_r_imu'])
        with self.assertRaises(eu.SuspectTrial) as caught:
            eu.load_trial('01', 'walking', dataset=self.DATASET, strict=True)
        self.assertIn('femur_r_imu', str(caught.exception))

    def test_strict_is_silent_on_a_clean_trial(self):
        self._set_suspect([])
        self.assertTrue(eu.load_trial('01', 'walking', dataset=self.DATASET, strict=True))

    def test_the_data_still_loads_when_it_warns(self):
        """A warning is not a refusal -- the plates must come back usable."""
        self._set_suspect(['femur_r_imu'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plates = eu.load_trial('01', 'walking', dataset=self.DATASET)
        self.assertEqual(len(plates), 2)


class TestRealTrialRoundTrip(unittest.TestCase):
    """End-to-end against real source data, if it is present."""

    def _live(self, subject, trial):
        """Straight from the dataset reader — what build_trials does."""
        if not paths.raw_trial_dir(subject, trial).is_dir():
            require_data(False, "no source data")
        return sources.get_source('alborno').load(subject, trial)

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


class TestQualityLabelling(unittest.TestCase):
    """The suspect flag and the accelerometer check, both of which were mis-specified.

    THE OLD SUSPECT FLAG WAS A SPEED DETECTOR. It compared an absolute residual against
    25 deg/s, and across 3303 plates that correlates with the plate's gyro RMS at r = +0.60:
    it flagged 94% of treadmill-running plates and 0% of static poses. Nobody believes running
    is almost always misaligned and a static pose never is. Normalizing by the plate's own
    signal drops the correlation to -0.31.
    """

    @staticmethod
    def _plate(name='p', residual_scale=0.0, signal_scale=1.0, seconds=20.0, rate=100.0):
        """A plate whose measured gyro is the mocap-derived one plus a known relative error."""
        from scipy.spatial.transform import Rotation
        from src.toolchest.IMUTrace import IMUTrace
        from src.toolchest.PlateTrial import PlateTrial
        from src.toolchest.WorldTrace import WorldTrace

        time = np.arange(int(seconds * rate)) / rate
        angles = signal_scale * np.column_stack([0.5 * np.sin(2 * np.pi * 0.8 * time),
                                                 0.3 * np.sin(2 * np.pi * 1.3 * time),
                                                 0.2 * np.sin(2 * np.pi * 0.5 * time)])
        world = WorldTrace(time, np.zeros((len(time), 3)),
                           Rotation.from_euler('zyx', angles).as_matrix())
        implied = world.calculate_imu_trace(skip_lin_acc=True)
        # A slow additive error, so it survives the low-pass the diagnostic applies.
        error = residual_scale * np.linalg.norm(implied.gyro, axis=1).mean() * np.column_stack(
            [np.sin(2 * np.pi * 0.3 * time)] * 3) / np.sqrt(3)
        imu = IMUTrace(time, implied.gyro + error, implied.acc, implied.mag)
        return PlateTrial(name, imu, world)

    def test_the_fraction_is_invariant_to_how_fast_the_plate_moved(self):
        """The whole point. The same RELATIVE error on a slow and a fast plate must score the
        same, where the absolute residual differs by the speed ratio."""
        slow = eu._alignment_residuals(self._plate(residual_scale=0.2, signal_scale=0.2))
        fast = eu._alignment_residuals(self._plate(residual_scale=0.2, signal_scale=2.0))
        self.assertAlmostEqual(slow['residual_fraction'], fast['residual_fraction'], places=2)
        # And the absolute figure, which is what the old rule used, does not agree.
        self.assertGreater(fast['gyro_residual_lowpass_rms_deg_s'],
                           5 * slow['gyro_residual_lowpass_rms_deg_s'])

    def test_a_clean_plate_scores_near_zero(self):
        out = eu._alignment_residuals(self._plate(residual_scale=0.0))
        self.assertLess(out['residual_fraction'], 0.05)

    def test_the_fraction_is_measured_in_the_same_band_as_the_residual(self):
        """Dividing a low-passed residual by a BROADBAND signal would flatter every fast
        trial, since the denominator picks up energy the numerator has had removed."""
        out = eu._alignment_residuals(self._plate(residual_scale=0.3))
        self.assertIn('gyro_signal_lowpass_rms_deg_s', out)
        self.assertLessEqual(out['gyro_signal_lowpass_rms_deg_s'],
                             out['gyro_signal_rms_deg_s'] * 1.01)

    def test_a_fast_plate_with_a_small_relative_error_is_not_suspect(self):
        """The regression this exists to prevent: 40 deg/s against a 200 deg/s signal is
        unremarkable, and the old rule called it suspect for being fast."""
        plates = {'fast': self._plate('fast', residual_scale=0.2, signal_scale=3.0)}
        self.assertEqual(eu.trial_diagnostics(plates)['suspect'], [])

    def test_a_genuinely_misaligned_plate_is_still_caught(self):
        plates = {'bad': self._plate('bad', residual_scale=1.2)}
        self.assertEqual(eu.trial_diagnostics(plates)['suspect'], ['bad'])


class TestStaticCalibration(unittest.TestCase):
    """`acc_norm_median` was being read against 9.81 as a scale check on every trial.

    On a dynamic trial that reading is simply wrong: all 159 plates more than 1 m/s^2 from
    gravity are treadmill running, topping out at 16.97, and a running limb genuinely spends
    more than half its time above 1 g. The statistic was answering a different question from
    the one asked of it.
    """

    @staticmethod
    def _plate(scale=1.0, moving=False, n=600):
        from src.toolchest.IMUTrace import IMUTrace
        from src.toolchest.PlateTrial import PlateTrial
        from src.toolchest.WorldTrace import WorldTrace

        time = np.arange(n) / 100.0
        gyro = np.full((n, 3), np.radians(60.0) if moving else 0.0)
        acc = np.tile([0.0, 0.0, scale * 9.80665], (n, 1))
        world = WorldTrace(time, np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1)))
        return PlateTrial('p', IMUTrace(time, gyro, acc, np.zeros((n, 3))), world)

    def test_a_still_plate_reports_its_scale(self):
        out = eu._static_calibration(self._plate(scale=1.0))
        self.assertAlmostEqual(out['acc_norm_static_median'], 9.80665, places=4)
        self.assertAlmostEqual(out['acc_scale_error'], 0.0, places=6)

    def test_a_real_scale_error_is_caught(self):
        out = eu._static_calibration(self._plate(scale=1.1))
        self.assertAlmostEqual(out['acc_scale_error'], 0.1, places=4)

    def test_a_moving_plate_reports_no_scale_at_all(self):
        """Silence rather than a wrong number. A treadmill-running plate carries no evidence
        about its own scale, and reporting one anyway is what produced 159 phantom faults."""
        out = eu._static_calibration(self._plate(scale=1.0, moving=True))
        self.assertNotIn('acc_norm_static_median', out)
        self.assertEqual(out['n_static_frames'], 0)

    def test_the_still_frames_are_chosen_on_the_gyro_not_the_accelerometer(self):
        """Selecting frames where |acc| is near 9.81 and then reporting that |acc| is near
        9.81 is circular, and would hide the very error the check looks for."""
        plate = self._plate(scale=1.5)
        out = eu._static_calibration(plate)
        self.assertAlmostEqual(out['acc_scale_error'], 0.5, places=4)


class TestFallbackOffsetIsVisible(unittest.TestCase):
    """Whether a plate's cluster-to-IMU offset was FITTED or DEFAULTED.

    A plate carries `sensor_offset` either way, so `sensor_offset_mm` in the manifest cannot
    be told apart from a nominal constant -- and on 10.4% of IMoVE's taped sensors it is one.
    Joint angles are untouched, because `shift_world_origin` moves positions and leaves
    rotations alone, but the lever-arm acceleration error is a median 0.04 m/s^2 and a p95 of
    0.30, so an acceleration comparison wants to be able to exclude those plates.
    """

    @staticmethod
    def _report(**per_plate):
        from src.toolchest.building.report import BuildReport
        report = BuildReport()
        for name, values in per_plate.items():
            report.add('S8_lever_arm', 'plate', name, **values)
        return report

    def test_a_defaulted_offset_is_marked_in_the_manifest(self):
        report = self._report(femur_r_imu={'used_fallback': True,
                                           'distance_from_nominal_mm': 0.0})
        plates = make_plates(names=('femur_r_imu',))
        stats = eu.trial_diagnostics(plates, report=report)['plates']['femur_r_imu']
        self.assertEqual(stats['sensor_offset_used_fallback'], 1.0)

    def test_a_fitted_offset_is_marked_too(self):
        """Absence of the flag would be ambiguous with 'this dataset has no lever-arm step',
        so the fitted case says so rather than staying silent."""
        report = self._report(femur_r_imu={'used_fallback': False,
                                           'distance_from_nominal_mm': 22.4})
        plates = make_plates(names=('femur_r_imu',))
        stats = eu.trial_diagnostics(plates, report=report)['plates']['femur_r_imu']
        self.assertEqual(stats['sensor_offset_used_fallback'], 0.0)
        self.assertAlmostEqual(stats['sensor_offset_distance_from_nominal_mm'], 22.4)

    def test_a_dataset_with_no_lever_arm_step_is_unaffected(self):
        """Al Borno has no cluster-to-IMU offset at all, and must not gain empty keys."""
        stats = eu.trial_diagnostics(make_plates(), report=None)['plates']['femur_r_imu']
        self.assertNotIn('sensor_offset_used_fallback', stats)
        self.assertIn('sensor_offset_mm', stats)

    def test_collecting_it_does_not_disturb_the_other_diagnostics(self):
        plates = make_plates()
        without = eu.trial_diagnostics(plates)
        with_report = eu.trial_diagnostics(plates, report=self._report())
        self.assertEqual(without, with_report)
