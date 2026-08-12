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
from experiments.experiment_utils import (DEFAULT_ACC_STD, DEFAULT_GYRO_STD, DEFAULT_MAG_STD,
                                          METHODS, STD_KEYS, pipeline_constants,
                                          resolve_method_spec, resolve_stds)
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
    def test_no_variant_is_the_flat_layout(self):
        path = paths.joint_angles_path('01', 'walking', 'mag_on')
        self.assertEqual(path, paths.JOINT_ANGLES_DIR / 'Subject01' / 'walking' / 'mag_on.parquet')

    def test_a_variant_inserts_one_level_under_the_joint_angles_root(self):
        path = paths.joint_angles_path('01', 'walking', 'mag_on', variant=VARIANT)
        self.assertEqual(path, paths.JOINT_ANGLES_DIR / VARIANT / 'Subject01' / 'walking' / 'mag_on.parquet')

    def test_a_variant_never_collides_with_the_default_tuning(self):
        """The one thing the variant exists for: same method name, same trial, two tunings.
        Without it the second run overwrites the first and the filename says nothing."""
        for method in DEFAULT_METHODS:
            with self.subTest(method=method):
                self.assertNotEqual(paths.joint_angles_path('01', 'walking', method),
                                    paths.joint_angles_path('01', 'walking', method, variant=VARIANT))

    def test_variants_do_not_collide_with_each_other(self):
        self.assertNotEqual(paths.joint_angles_path('01', 'walking', 'mag_on', variant='a'),
                            paths.joint_angles_path('01', 'walking', 'mag_on', variant='b'))

    def test_a_variant_path_is_still_outside_the_read_only_data_tree(self):
        paths.ensure_parent(paths.joint_angles_path('01', 'walking', 'mag_on', variant=VARIANT))


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
