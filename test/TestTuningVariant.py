"""
Covers the filter-tuning override and the output namespacing it requires:
experiment_utils.resolve_stds / pipeline_constants, paths.joint_angles_path's `variant`,
and the configuration experiments/normalized_benchmark.py builds on top of both.

Every failure mode here is silent. An unknown std key dropped on the floor runs the
default tuning under a name that claims otherwise; a variant that does not reach the
output path overwrites the benchmark's parquets in place; a manifest that reports
DEFAULT_ACC_STD next to a re-tuned run makes the provenance sidecar actively misleading.
None of it raises, and all of it produces plausible numbers.

Pure configuration and path arithmetic, so no data and no filter runs — fast, always
runnable.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import paths
from experiments.experiment_utils import (DATASET_STDS, DEFAULT_ACC_STD, DEFAULT_GYRO_STD,
                                          DEFAULT_MAG_STD, METHODS, STD_KEYS, TRIAL_DATASET,
                                          pipeline_constants, resolve_method_spec, resolve_stds)
from experiments.normalized_benchmark import (BASE_METHODS, DEFAULT_METHODS, NORMALIZED_METHODS,
                                              STATS_NAME, TUNED_STDS, VARIANT, base_of)


class TestResolveStds(unittest.TestCase):
    def test_no_override_is_the_shipped_tuning(self):
        self.assertEqual(resolve_stds(), {'gyro_std': DEFAULT_GYRO_STD,
                                          'acc_std': DEFAULT_ACC_STD,
                                          'mag_std': DEFAULT_MAG_STD})
        self.assertEqual(resolve_stds(None), resolve_stds())
        self.assertEqual(resolve_stds({}), resolve_stds())

    def test_a_partial_override_leaves_the_other_sensors_alone(self):
        resolved = resolve_stds({'acc_std': 0.03})
        self.assertEqual(resolved['acc_std'], 0.03)
        self.assertEqual(resolved['gyro_std'], DEFAULT_GYRO_STD)
        self.assertEqual(resolved['mag_std'], DEFAULT_MAG_STD)

    def test_every_key_can_be_overridden(self):
        override = {'gyro_std': 1.0, 'acc_std': 2.0, 'mag_std': 3.0}
        self.assertEqual(resolve_stds(override), override)

    def test_an_unknown_key_raises_and_names_it(self):
        """The whole point of resolving through one function: 'acc_stdev' silently ignored
        would run the default accelerometer weighting under a re-tuned label."""
        for bad in ({'acc_stdev': 0.03}, {'mag': 0.05}, {'acc_std': 0.03, 'gyro': 0.01}):
            with self.subTest(stds=bad):
                with self.assertRaises(ValueError) as ctx:
                    resolve_stds(bad)
                self.assertIn('Unknown filter std key', str(ctx.exception))

    def test_values_are_coerced_to_float(self):
        """argparse gives floats, but a hand-written dict of ints would otherwise reach
        np.ones(3) * std as an int array."""
        resolved = resolve_stds({'acc_std': 1, 'gyro_std': '0.5'})
        self.assertIsInstance(resolved['acc_std'], float)
        self.assertEqual(resolved['gyro_std'], 0.5)

    def test_the_result_is_a_fresh_dict(self):
        """Callers thread it into manifests and filter calls; a shared dict would let one
        run's mutation retune the next."""
        first = resolve_stds({'acc_std': 0.03})
        first['acc_std'] = 99.0
        self.assertEqual(resolve_stds({'acc_std': 0.03})['acc_std'], 0.03)

    def test_std_keys_matches_what_resolve_stds_returns(self):
        self.assertEqual(set(STD_KEYS), set(resolve_stds()))


class TestDatasetStds(unittest.TestCase):
    """The tuning is per dataset (see experiments/filter_gains.py). Every failure here is the
    same silent one: a run reads another lab's magnetometer weighting and says nothing."""

    def test_the_bare_defaults_are_the_default_dataset_row(self):
        """DEFAULT_*_STD still exist for the dozen call sites that read them directly. If
        they ever drift from DATASET_STDS[TRIAL_DATASET], half the pipeline runs one tuning
        and half runs another."""
        self.assertEqual(
            {'gyro_std': DEFAULT_GYRO_STD, 'acc_std': DEFAULT_ACC_STD, 'mag_std': DEFAULT_MAG_STD},
            DATASET_STDS[TRIAL_DATASET])

    def test_each_dataset_resolves_to_its_own_row(self):
        for dataset, row in DATASET_STDS.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(resolve_stds(dataset=dataset), row)

    def test_the_datasets_are_not_all_the_same_tuning(self):
        """If they were, the per-dataset table would be a more complicated way to write one
        triple, and this whole mechanism should be deleted rather than believed."""
        distinct = {tuple(sorted(row.items())) for row in DATASET_STDS.values()}
        self.assertGreater(len(distinct), 1)

    def test_an_unknown_dataset_raises_instead_of_falling_back(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_stds(dataset='not_a_dataset')
        self.assertIn('not_a_dataset', str(ctx.exception))
        self.assertIn('filter_gains', str(ctx.exception))

    def test_every_registered_dataset_has_a_tuning(self):
        """The registry is the list of things that can be tracked; a dataset in it with no
        row cannot run at all now that the fallback raises."""
        from experiments.global_assumptions import DATASETS, tracking_spec
        for name in DATASETS:
            with self.subTest(dataset=name):
                self.assertIn(tracking_spec(name).dataset, DATASET_STDS)

    def test_an_override_still_beats_the_dataset_row(self):
        resolved = resolve_stds({'acc_std': 0.03}, dataset='imove')
        self.assertEqual(resolved['acc_std'], 0.03)
        self.assertEqual(resolved['mag_std'], DATASET_STDS['imove']['mag_std'])

    def test_pipeline_constants_records_which_row_it_used(self):
        """Three stds with no note of which dataset they came from cannot be checked against
        anything, now that there is more than one row."""
        constants = pipeline_constants(dataset='imove')
        self.assertEqual(constants['stds_dataset'], 'imove')
        self.assertEqual(constants['mag_std'], DATASET_STDS['imove']['mag_std'])


class TestPipelineConstants(unittest.TestCase):
    def test_no_override_reports_the_defaults(self):
        constants = pipeline_constants()
        self.assertEqual(constants['gyro_std'], DEFAULT_GYRO_STD)
        self.assertEqual(constants['acc_std'], DEFAULT_ACC_STD)
        self.assertEqual(constants['mag_std'], DEFAULT_MAG_STD)

    def test_an_override_is_what_gets_recorded(self):
        """A manifest is the only record of which tuning produced the file beside it, so
        reporting the default next to a re-tuned run is worse than reporting nothing."""
        constants = pipeline_constants(TUNED_STDS)
        for key, value in TUNED_STDS.items():
            self.assertEqual(constants[key], value)

    def test_the_other_constants_survive_an_override(self):
        constants = pipeline_constants(TUNED_STDS)
        self.assertIn('expected_gravity', constants)
        self.assertIn('mag_adapt_threshold', constants)

    def test_an_unknown_key_raises_here_too(self):
        with self.assertRaises(ValueError):
            pipeline_constants({'acc_stdev': 0.03})


class TestJointAnglesVariant(unittest.TestCase):
    def test_no_variant_is_the_dataset_namespaced_layout(self):
        path = paths.joint_angles_path('alborno', '01', 'walking', 'mag_on')
        self.assertEqual(path,
                         paths.JOINT_ANGLES_DIR / 'alborno' / '01' / 'walking' / 'mag_on.parquet')

    def test_a_variant_inserts_one_level_under_the_joint_angles_root(self):
        path = paths.joint_angles_path('alborno', '01', 'walking', 'mag_on', variant=VARIANT)
        self.assertEqual(path, paths.JOINT_ANGLES_DIR / VARIANT / 'alborno' / '01' / 'walking'
                         / 'mag_on.parquet')

    def test_a_variant_never_collides_with_the_default_tuning(self):
        """The one thing the variant exists for: same method name, same trial, two tunings.
        Without it the second run overwrites the first and the filename says nothing."""
        for method in DEFAULT_METHODS:
            with self.subTest(method=method):
                self.assertNotEqual(
                    paths.joint_angles_path('alborno', '01', 'walking', method),
                    paths.joint_angles_path('alborno', '01', 'walking', method, variant=VARIANT))

    def test_variants_do_not_collide_with_each_other(self):
        self.assertNotEqual(paths.joint_angles_path('alborno', '01', 'walking', 'mag_on',
                                                    variant='a'),
                            paths.joint_angles_path('alborno', '01', 'walking', 'mag_on',
                                                    variant='b'))

    def test_datasets_do_not_collide_with_each_other(self):
        """The same guard one dimension out. Two datasets carry the same trial names and the
        same method names, so without the namespace an IMoVE run overwrites Al Borno's
        parquets under a path that claims to be Al Borno's."""
        self.assertNotEqual(paths.joint_angles_path('alborno', '01', 'walking', 'mag_on'),
                            paths.joint_angles_path('imove', '01', 'walking', 'mag_on'))

    def test_a_trial_key_with_slashes_stays_under_its_subject(self):
        """The biplane half names a trial by the session and block it came from, so the key
        contains separators. It has to land under the subject directory, at the same depth the
        build tree puts it."""
        path = paths.joint_angles_path('imove_biplane', '12', 'Test1/A/RSDrop1', 'mag_off')
        self.assertEqual(path, paths.JOINT_ANGLES_DIR / 'imove_biplane' / '12' / 'Test1' / 'A'
                         / 'RSDrop1' / 'mag_off.parquet')

    def test_a_variant_path_is_still_outside_the_read_only_data_tree(self):
        paths.ensure_parent(
            paths.joint_angles_path('alborno', '01', 'walking', 'mag_on', variant=VARIANT))


class TestNormalizedBenchmarkConfig(unittest.TestCase):
    def test_the_requested_tuning_is_what_is_configured(self):
        """The numbers this experiment was created for. Pinned so a later edit to the
        DEFAULT_* constants or to this dict is a test failure rather than a quiet change
        to what 'normalized_benchmark' means."""
        self.assertEqual(TUNED_STDS, {'gyro_std': 0.0116, 'acc_std': 0.03, 'mag_std': 0.05})

    def test_every_arm_normalizes(self):
        """The other half of the experiment's definition. The suffix is an override, so
        this holds for the three relative-filter bases (default off) and the EKF (on)
        alike."""
        for method in NORMALIZED_METHODS:
            with self.subTest(method=method):
                self.assertTrue(resolve_method_spec(method)['normalize_measurements'])

    def test_no_arm_rescales_its_stds(self):
        """'_normalized', not '_rescaled': the configured stds are taken at face value
        against unit-length measurements. Rescaling would divide them by each sensor's
        nominal magnitude and run a different tuning than the one asked for."""
        for method in NORMALIZED_METHODS:
            with self.subTest(method=method):
                self.assertFalse(resolve_method_spec(method)['rescale_stds'])

    def test_the_arms_are_the_benchmark_grid_minus_the_ground_truth(self):
        self.assertEqual(set(BASE_METHODS), set(METHODS) - {'marker'})

    def test_the_method_list_carries_the_ground_truth(self):
        """compute_error_stats has nothing to compare against without it, and returns an
        empty frame rather than failing."""
        self.assertIn('marker', DEFAULT_METHODS)
        self.assertEqual(DEFAULT_METHODS, ['marker'] + NORMALIZED_METHODS)

    def test_every_arm_resolves(self):
        for method in DEFAULT_METHODS:
            with self.subTest(method=method):
                resolve_method_spec(method)

    def test_each_arm_differs_from_its_base_only_in_the_normalization_flags(self):
        for base, method in zip(BASE_METHODS, NORMALIZED_METHODS):
            with self.subTest(method=method):
                spec = resolve_method_spec(method)
                for key, value in METHODS[base].items():
                    if key in ('normalize_measurements', 'rescale_stds'):
                        continue
                    self.assertEqual(spec[key], value)

    def test_base_of_inverts_the_name_construction(self):
        """The printed summary joins this run's arms to the benchmark's by base name; a
        mismatch would compare an arm against the wrong reference method."""
        for base, method in zip(BASE_METHODS, NORMALIZED_METHODS):
            with self.subTest(method=method):
                self.assertEqual(base_of(method), base)

    def test_base_of_leaves_an_unsuffixed_name_alone(self):
        self.assertEqual(base_of('marker'), 'marker')
        self.assertEqual(base_of('mag_on'), 'mag_on')

    def test_the_statistics_file_is_not_the_benchmarks(self):
        self.assertNotEqual(paths.statistics_path(STATS_NAME), paths.statistics_path('all_subject'))


if __name__ == '__main__':
    unittest.main()
