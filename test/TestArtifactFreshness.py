"""Every layer refuses an artifact it cannot account for.

`load_trial` has refused a stale parquet for as long as there has been a trial cache. The two
layers built ON TOP of it — the joint angles, and the per-trial statistics pooled from them —
refused nothing: they read whatever was on disk and stamped the summary with the tuning the
command line asked for, so a run could report numbers produced by code and a filter that no
longer existed and nothing in the output would say so. That is not a hypothetical; it happened,
at a 28x difference in magnetometer trust, and the conclusion drawn from it was an artifact.

These tests are about the refusal, not about the physics. They build synthetic trees so each
check can be triggered on its own — a moved tuning, an edited estimator, a rebuilt trial, a
truncated file — because the value of a fail-closed cache is entirely in which of those it
notices, and a test that only ever sees a healthy tree cannot tell.
"""
import json
import os
import pathlib
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

os.environ.setdefault("DISABLE_TQDM", "True")

import paths
from experiments import experiment_utils as eu


def angles_frame(num_samples: int = 8) -> pd.DataFrame:
    """A joint-angle table shaped like the real one, small enough to be free."""
    return pd.DataFrame({
        'timestamp': np.arange(num_samples) / 100.0,
        'joint_name': ['R_Knee'] * num_samples,
        'rx': np.linspace(0.0, 0.1, num_samples),
        'ry': np.zeros(num_samples),
        'rz': np.zeros(num_samples),
    })


class FreshnessFixture(unittest.TestCase):
    """A results tree of its own, and a source trial that is fresh by assertion.

    `_read_side_trial_state` is stubbed rather than satisfied. Satisfying it would mean a real
    source directory, a real inventory and a real parquet — the trial layer's own tests already
    cover all of that, and every test here would then be testing it again on the way past. What
    these tests need from it is a knob: one test turns it to 'stale' precisely to check that a
    bad trial is reported before anything about the angles.
    """

    TRIAL_KEY = {'schema_version': 7, 'toolchest_digest': 'aaaaaaaaaaaaaaaa',
                 'sources': [{'name': 'walking.trc', 'bytes': 123}], 'n_rows': 100, 'n_plates': 2}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = pathlib.Path(self.tmp.name)

        for name, sub in (('JOINT_ANGLES_DIR', 'joint_angles'), ('EXPERIMENTS_DIR', 'experiments'),
                          ('TRIALS_DIR', 'trials'), ('PER_SUBJECT_STATS_DIR', 'per_subject')):
            patch = mock.patch.object(paths, name, root / sub)
            patch.start()
            self.addCleanup(patch.stop)

        trial_patch = mock.patch.object(eu, '_read_side_trial_state',
                                        return_value=('fresh', None, None))
        self.trial_state = trial_patch.start()
        self.addCleanup(trial_patch.stop)

        # The sidecar `_source_trial_key` identifies the build by. Only the manifest is needed:
        # whether the parquet itself is usable is the trial layer's question, and it has been
        # answered by the stub above.
        self.write_trial_manifest(self.TRIAL_KEY)

        # No test here opts out, so a leaked variable from another one cannot make these pass.
        self.addCleanup(os.environ.pop, eu.ALLOW_STALE_ENV, None)
        os.environ.pop(eu.ALLOW_STALE_ENV, None)

    def write_trial_manifest(self, cache_key):
        path = paths.cached_trial_path('alborno', '01', 'walking')
        paths.ensure_parent(path)
        paths.write_manifest(path, cache_key=cache_key, dataset='alborno')

    def write_angles(self, method='mag_on', subject='01', trial='walking', stds=None,
                     variant=None, experiment=paths.BENCHMARK_EXPERIMENT, df=None):
        eu.save_joint_angles(angles_frame() if df is None else df, 'alborno', subject, trial,
                             method, experiment=experiment, variant=variant, stds=stds)
        return paths.joint_angles_path('alborno', subject, trial, method, variant=variant,
                                       experiment=None if experiment == paths.BENCHMARK_EXPERIMENT
                                       else experiment)

    def edit_manifest(self, path, **changes):
        """Rewrites fields in an artifact's sidecar, to stand in for whatever produced them."""
        target = paths.manifest_path(path)
        manifest = json.loads(target.read_text())
        manifest.update(changes)
        target.write_text(json.dumps(manifest, indent=2, default=str) + '\n')

    def drop_from_manifest(self, path, *fields):
        target = paths.manifest_path(path)
        manifest = json.loads(target.read_text())
        for field in fields:
            manifest.pop(field, None)
        target.write_text(json.dumps(manifest, indent=2, default=str) + '\n')

    def status(self, **kwargs):
        return eu.joint_angles_status('alborno', kwargs.pop('subject', '01'),
                                      kwargs.pop('trial', 'walking'),
                                      kwargs.pop('method', 'mag_on'), **kwargs)


class TestAHealthyArtifactIsServed(FreshnessFixture):
    """The check has to be silent on a good tree, or it is not a check, it is an obstacle."""

    def test_what_this_run_just_wrote_is_fresh(self):
        self.write_angles()
        self.assertEqual(self.status(), ('fresh', None))

    def test_it_loads(self):
        self.write_angles()
        df = eu.load_joint_angles('alborno', '01', 'walking', 'mag_on')
        self.assertEqual(len(df), len(angles_frame()))

    def test_a_retuned_arm_is_fresh_against_its_own_tuning(self):
        """`stds` travels with `variant` on the read side too. A caller that namespaced its
        output under a re-tuning and then checked it against the module default would call every
        arm it just wrote stale — which is the false alarm that gets a check switched off."""
        stds = {'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03}
        self.write_angles(variant='retuned', stds=stds)
        self.assertEqual(self.status(variant='retuned', stds=stds), ('fresh', None))

    def test_never_written_is_missing_not_stale(self):
        """A partial run is a legitimate state. Only a file that contradicts its own manifest
        is not, and conflating the two would make every incomplete sweep unreadable."""
        self.assertEqual(self.status(), ('missing', None))
        self.assertIsNone(eu.load_joint_angles('alborno', '01', 'walking', 'mag_on'))


class TestTheTuningIsChecked(FreshnessFixture):
    """The failure this whole layer exists for: angles from one filter, a summary claiming
    another. The sidecars recorded the constants all along; nothing compared them."""

    def test_a_moved_tuning_is_refused(self):
        self.write_angles(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03})
        status, reason = self.status(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.9})
        self.assertEqual(status, 'stale')
        self.assertIn('constants', reason)

    def test_the_refusal_is_an_exception_not_a_return_value(self):
        self.write_angles(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03})
        with self.assertRaises(eu.StaleJointAngles) as caught:
            eu.load_joint_angles('alborno', '01', 'walking', 'mag_on',
                                 stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.9})
        self.assertIn('Fix:', str(caught.exception))

    def test_a_float_round_trip_is_not_a_mismatch(self):
        """The tuning is compared numerically. A ladder rung recomputed to the last bit
        differently is not a different filter, and reporting it as one would be noise."""
        stds = {'gyro_std': 0.0116, 'acc_std': 0.03, 'mag_std': 0.05}
        path = self.write_angles(stds=stds)
        nudged = dict(json.loads(paths.manifest_path(path).read_text())['constants'])
        nudged['acc_std'] = nudged['acc_std'] * (1 + 1e-13)
        self.edit_manifest(path, constants=nudged)
        self.assertEqual(self.status(stds=stds), ('fresh', None))


class TestTheCodeIsChecked(FreshnessFixture):
    def test_an_edited_estimator_is_refused(self):
        """The trial digest cannot see this. It hashes what turns files into a PlateTrial, which
        is upstream of every filter — editing the relative filter's update step changed every
        joint angle in the tree and moved no digest at all."""
        self.write_angles()
        with mock.patch.object(eu, '_estimator_digest', return_value='ffffffffffffffff'):
            status, reason = self.status()
        self.assertEqual(status, 'stale')
        self.assertIn('filter code has changed', reason)

    def test_a_redefined_method_is_refused(self):
        """The method NAME is in the path; what it means is in `METHODS`. Changing what
        'mag_on' does leaves every filename identical and every parquet wrong."""
        self.write_angles(method='mag_on')
        redefined = dict(eu.METHODS)
        redefined['mag_on'] = {**redefined['mag_on'], 'mag_mode': 'off'}
        with mock.patch.object(eu, 'METHODS', redefined):
            eu.resolve_method_spec.cache_clear() if hasattr(
                eu.resolve_method_spec, 'cache_clear') else None
            status, reason = self.status()
        self.assertEqual(status, 'stale')
        self.assertIn('method_spec', reason)


class TestTheTrialUnderneathIsChecked(FreshnessFixture):
    def test_a_stale_trial_makes_its_angles_stale(self):
        """Reported before anything about the angles themselves, because rebuilding the trial is
        the one action that clears every arm over it at once."""
        self.write_angles()
        self.trial_state.return_value = ('stale', 'toolchest_digest moved', None)
        status, reason = self.status()
        self.assertEqual(status, 'stale')
        self.assertIn('cached trial', reason)
        self.assertIn('toolchest_digest moved', reason)

    def test_a_rebuilt_trial_makes_its_angles_stale(self):
        """The trial's own status cannot catch this: a rebuild leaves it FRESH again, under new
        content, while the angles beside it still describe the plates it used to hold."""
        self.write_angles()
        self.write_trial_manifest({**self.TRIAL_KEY, 'toolchest_digest': 'bbbbbbbbbbbbbbbb'})
        status, reason = self.status()
        self.assertEqual(status, 'stale')
        self.assertIn('rebuilt', reason)


class TestTheArtifactItselfIsChecked(FreshnessFixture):
    def test_a_truncated_parquet_is_refused(self):
        """Every other comparison is manifest-against-code and a short file passes them all. A
        --force run interrupted between the parquet and the sidecar leaves exactly this."""
        path = self.write_angles()
        self.edit_manifest(path, n_rows=999999)
        status, reason = self.status()
        self.assertEqual(status, 'stale')
        self.assertIn('truncated', reason)

    def test_a_manifest_written_before_this_check_is_refused(self):
        """7000 parquets on disk predate these fields. They are not wrong, they are
        unverifiable, and the report says so rather than naming an arbitrary missing key."""
        path = self.write_angles()
        self.drop_from_manifest(path, 'artifact_key', 'source_key', 'estimator_digest')
        status, reason = self.status()
        self.assertEqual(status, 'stale')
        self.assertIn('before the joint-angle freshness check', reason)

    def test_no_sidecar_at_all_is_refused(self):
        path = self.write_angles()
        paths.manifest_path(path).unlink()
        self.assertEqual(self.status()[0], 'stale')


class TestTheWholeGridIsReportedAtOnce(FreshnessFixture):
    """A run whose tuning moved has every arm of every trial stale. Naming one of them invites
    fixing them one at a time."""

    ROWS = [('01', 'walking'), ('01', 'complexTasks')]
    METHODS = ['mag_on', 'mag_off']

    def write_grid(self, stds=None):
        for subject, trial in self.ROWS:
            for method in self.METHODS:
                self.write_angles(method=method, subject=subject, trial=trial, stds=stds)

    def test_every_stale_arm_is_named_in_one_error(self):
        self.write_grid(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03})
        with self.assertRaises(eu.StaleJointAngles) as caught:
            eu.load_all_joint_angles('alborno', self.ROWS, self.METHODS,
                                     stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.9})
        message = str(caught.exception)
        self.assertIn('4 artifact(s)', message)
        for subject, trial in self.ROWS:
            for method in self.METHODS:
                self.assertIn(f'alborno/{subject}/{trial}/{method}', message)

    def test_it_does_not_read_a_gigabyte_before_refusing(self):
        """Status first across the whole grid, then load. The check costs footer reads; being
        wrong about it should not cost the concatenation."""
        self.write_grid(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03})
        with mock.patch.object(eu.pd, 'read_parquet') as read:
            with self.assertRaises(eu.StaleJointAngles):
                eu.load_all_joint_angles('alborno', self.ROWS, self.METHODS,
                                         stds={'gyro_std': 0.9, 'acc_std': 0.9, 'mag_std': 0.9})
        read.assert_not_called()

    def test_a_healthy_grid_pools(self):
        self.write_grid()
        df = eu.load_all_joint_angles('alborno', self.ROWS, self.METHODS)
        self.assertEqual(len(df), len(angles_frame()) * 4)
        self.assertEqual(set(df['method'].unique()), set(self.METHODS))

    def test_an_absent_arm_warns_rather_than_vanishing(self):
        """This message used to be a print gated on DISABLE_TQDM — which this module sets to
        'True' at import, so it could only ever appear in a process that had overridden it
        again. A pooled summary could silently be short an entire arm."""
        self.write_angles(method='mag_on')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            df = eu.load_all_joint_angles('alborno', [('01', 'walking')], ['mag_on', 'mag_off'])
        self.assertEqual(set(df['method'].unique()), {'mag_on'})
        self.assertEqual([w.category for w in caught], [eu.MissingJointAnglesWarning])
        self.assertIn('mag_off', str(caught[0].message))


class TestTheOptOut(FreshnessFixture):
    """One escape hatch, in the environment, for the one legitimate case: looking at what an old
    run produced, on purpose. It has to survive a process boundary — the statistics phase reads
    inside spawned workers — and it has to be loud on every use."""

    def test_it_warns_and_serves_instead_of_raising(self):
        self.write_angles(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03})
        os.environ[eu.ALLOW_STALE_ENV] = '1'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            df = eu.load_joint_angles('alborno', '01', 'walking', 'mag_on',
                                      stds={'gyro_std': 0.9, 'acc_std': 0.9, 'mag_std': 0.9})
        self.assertEqual(len(df), len(angles_frame()))
        self.assertEqual([w.category for w in caught], [eu.StaleArtifactWarning])

    def test_it_is_off_unless_explicitly_set(self):
        for value in ('', '0', 'no', 'false'):
            os.environ[eu.ALLOW_STALE_ENV] = value
            self.assertFalse(eu.stale_artifacts_allowed(), value)
        for value in ('1', 'true', 'YES', 'on'):
            os.environ[eu.ALLOW_STALE_ENV] = value
            self.assertTrue(eu.stale_artifacts_allowed(), value)

    def test_a_caller_can_still_refuse_regardless(self):
        """`allow_stale=False` outranks the environment, so a script that must not run on stale
        data cannot be talked into it by an exported variable."""
        self.write_angles(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03})
        os.environ[eu.ALLOW_STALE_ENV] = '1'
        with self.assertRaises(eu.StaleJointAngles):
            eu.load_joint_angles('alborno', '01', 'walking', 'mag_on', allow_stale=False,
                                 stds={'gyro_std': 0.9, 'acc_std': 0.9, 'mag_std': 0.9})


class TestTheStatisticsLayer(FreshnessFixture):
    """The layer the aggregation phase reads. Its hazard is specific: a trial whose statistics
    worker FAILED this run still has last run's table on disk, and the summary pools it while
    claiming, from the command line, to describe this run."""

    STATS = 'all_subject'

    def write_stats(self, methods=('mag_on', 'mag_off'), variant=None, stds=None,
                    anatomical_axes=False, subject='01', trial='walking'):
        for method in methods:
            self.write_angles(method=method, subject=subject, trial=trial, variant=variant,
                              stds=stds)
        path = paths.ensure_parent(
            paths.per_subject_statistics_path(self.STATS, 'alborno', subject, trial))
        frame = pd.DataFrame({'joint_name': ['R_Knee'], 'method': ['mag_on'], 'rmse': [1.0]})
        frame.to_parquet(path, engine='pyarrow')
        paths.write_manifest(
            path, constants=eu.pipeline_constants(stds, dataset='alborno'),
            experiment=self.STATS, dataset='alborno', subject=subject, trial=trial,
            methods=sorted(methods), variant=variant, anatomical_axes=anatomical_axes,
            n_rows=len(frame),
            angles_key=eu.pooled_angles_key('alborno', subject, trial, methods, variant=variant))
        return path

    def test_a_healthy_table_is_fresh(self):
        self.write_stats()
        self.assertEqual(
            eu.per_trial_statistics_status(self.STATS, 'alborno', '01', 'walking',
                                           methods=['mag_on', 'mag_off']),
            ('fresh', None))

    def test_regenerated_angles_invalidate_the_table_that_pooled_them(self):
        """The check the recorded constants cannot make. An edit to the filter itself moves the
        angles without moving the tuning, and the table looked perfectly current."""
        self.write_stats()
        self.write_angles(method='mag_on', df=angles_frame(num_samples=9))
        status, reason = eu.per_trial_statistics_status(self.STATS, 'alborno', '01', 'walking',
                                                        methods=['mag_on', 'mag_off'])
        self.assertEqual(status, 'stale')
        self.assertIn('regenerated', reason)

    def test_a_table_pooled_from_other_arms_is_refused(self):
        """Half the arms of a comparison is not a comparison, however internally consistent the
        table is."""
        self.write_stats(methods=('mag_on',))
        status, reason = eu.per_trial_statistics_status(self.STATS, 'alborno', '01', 'walking',
                                                        methods=['mag_on', 'mag_off'])
        self.assertEqual(status, 'stale')
        self.assertIn('methods', reason)

    def test_a_table_without_the_requested_axes_is_refused(self):
        """A table missing an axis for some trials and not others pools into a figure whose
        panels are drawn from different subject sets."""
        self.write_stats(anatomical_axes=False)
        status, reason = eu.per_trial_statistics_status(
            self.STATS, 'alborno', '01', 'walking', methods=['mag_on', 'mag_off'],
            anatomical_axes=True)
        self.assertEqual(status, 'stale')
        self.assertIn('anatomical', reason)

    def test_a_table_from_another_tuning_is_refused(self):
        self.write_stats(stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.03})
        status, reason = eu.per_trial_statistics_status(
            self.STATS, 'alborno', '01', 'walking', methods=['mag_on', 'mag_off'],
            stds={'gyro_std': 0.01, 'acc_std': 0.02, 'mag_std': 0.9})
        self.assertEqual(status, 'stale')
        self.assertIn('constants', reason)

    def test_the_loader_raises_and_names_every_trial(self):
        self.write_stats(subject='01', trial='walking', methods=('mag_on',))
        self.write_stats(subject='02', trial='walking', methods=('mag_on',))
        with self.assertRaises(eu.StaleStatistics) as caught:
            eu.load_per_trial_statistics(self.STATS, 'alborno',
                                         [('01', 'walking'), ('02', 'walking')],
                                         methods=['mag_on', 'mag_off'])
        self.assertIn('2 artifact(s)', str(caught.exception))

    def test_a_trial_with_no_table_is_still_skipped_quietly(self):
        """A partial run stays legitimate at this layer too — the caller reports what it found,
        and the benchmark's shortfall report is what names the gaps."""
        self.write_stats()
        df = eu.load_per_trial_statistics(self.STATS, 'alborno',
                                          [('01', 'walking'), ('09', 'walking')],
                                          methods=['mag_on', 'mag_off'])
        self.assertEqual(len(df), 1)


class TestTheStatisticsWorkerRecordsWhatItPooled(FreshnessFixture):
    def test_the_manifest_names_the_arms_that_arrived_not_the_ones_requested(self):
        """It recorded the REQUEST, so a table built from three arms of a five-arm run claimed
        all five — the one thing its reader most needs to know, actively hidden."""
        self.write_angles(method='mag_on')
        state = {}
        with mock.patch.object(eu, 'compute_error_stats',
                              return_value=pd.DataFrame({'joint_name': ['R_Knee'],
                                                         'method': ['mag_on'], 'rmse': [1.0]})):
            eu.compute_stats_worker(('01', 'walking'), ['stats'], state,
                                    methods=['mag_on', 'mag_off'], stats_name='probe',
                                    experiment=paths.BENCHMARK_EXPERIMENT)
        self.assertEqual(state[(('01', 'walking'), 'stats')], 'Success')
        manifest = paths.read_manifest(
            paths.per_subject_statistics_path('probe', 'alborno', '01', 'walking'))
        self.assertEqual(manifest['methods'], ['mag_on'])
        self.assertEqual(manifest['requested_methods'], ['mag_off', 'mag_on'])

    def test_a_stale_arm_fails_the_cell_rather_than_being_left_out(self):
        """The worker catches everything and reports it in the grid, so a refusal surfaces as a
        red cell with the reason in it instead of a table quietly short an arm."""
        self.write_angles(method='mag_on', stds={'gyro_std': 0.01, 'acc_std': 0.02,
                                                 'mag_std': 0.03})
        state = {}
        eu.compute_stats_worker(('01', 'walking'), ['stats'], state, methods=['mag_on'],
                                stats_name='probe', experiment=paths.BENCHMARK_EXPERIMENT,
                                stds={'gyro_std': 0.9, 'acc_std': 0.9, 'mag_std': 0.9})
        self.assertIn('Failed', state[(('01', 'walking'), 'stats')])
        self.assertIn('stale', state[(('01', 'walking'), 'stats')])


class TestOneNameCatchesEveryLayer(unittest.TestCase):
    def test_all_three_share_a_base(self):
        """So a caller that genuinely wants to survive any staleness catches one name, and so
        the rule reads the same at every layer."""
        for error in (eu.StaleTrialCache, eu.StaleJointAngles, eu.StaleStatistics):
            self.assertTrue(issubclass(error, eu.StaleArtifact), error.__name__)
        self.assertTrue(issubclass(eu.StaleArtifact, RuntimeError))


if __name__ == '__main__':
    unittest.main()
