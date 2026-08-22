"""
Covers experiment_utils.TrackingSpec and global_assumptions.tracking_spec — what the
benchmark pipeline knows about a dataset before it runs a filter over it.

Four things used to be Al Borno constants and are now per-dataset: the joint table, the
world-frame gravity vector, the sensor the magnetometer oracle is referenced against, and
whether there is a magnetometer at all. EVERY ONE OF THEM FAILS SILENTLY IF IT IS WRONG:

  * a joint table naming absent sensors produces no joints, not an error
  * a wrong gravity axis corrupts the EKF's world reference and both acc oracles, and
    nothing else in the run looks different
  * a wrong mag reference produces a plausible field from the wrong place on the body
  * a missing magnetometer produces mag_off's answer under mag_on's name, because the
    filter drops a zero-length vector measurement out of its update

So these tests pin the resolved values themselves rather than only the plumbing, and the
Al Borno spec is pinned against the module constants it replaced — that equality is the
guarantee that this refactor changed no existing number.

Fixture plates only, no built trials, so this is fast and always runnable.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import (ALBORNO_TRACKING, EXPECTED_GRAVITY, JOINTS,
                                         TRIAL_DATASET, TrackingSpec,
                                         _compute_expected_mag_field,
                                         _setup_ekf_ground_plate_, compute_joint_angles,
                                         trial_task)
from experiments.benchmark_experiment import unscored_trials
from experiments.global_assumptions import DATASETS, get_dataset, tracking_spec
from test.TestExperimentPhysics import WORLD_MAG, make_plate


class TestAlbornoIsUnchanged(unittest.TestCase):
    """The resolved Al Borno spec must be the constants it was extracted from.

    If this drifts, every result produced before the dataset dimension existed was computed
    under a different configuration than a re-run would use, and no manifest would say so.
    """

    def setUp(self):
        self.resolved = tracking_spec('alborno')

    def test_the_joint_table_is_the_module_constant(self):
        self.assertEqual(self.resolved.joints, dict(JOINTS))

    def test_gravity_is_the_module_constant(self):
        np.testing.assert_allclose(self.resolved.gravity, EXPECTED_GRAVITY)

    def test_the_mag_reference_is_the_torso(self):
        """_compute_expected_mag_field selected plates by the substring 'torso' before this
        was a spec field, and 'torso_imu' is the only plate that matched."""
        self.assertEqual(self.resolved.mag_reference, 'torso_imu')

    def test_it_matches_the_module_default_used_by_every_older_caller(self):
        for field in ('joints', 'mag_reference', 'has_magnetometer', 'subject_label'):
            with self.subTest(field=field):
                self.assertEqual(getattr(self.resolved, field),
                                 getattr(ALBORNO_TRACKING, field))
        np.testing.assert_allclose(self.resolved.gravity, ALBORNO_TRACKING.gravity)

    def test_the_default_is_alborno(self):
        self.assertEqual(ALBORNO_TRACKING.dataset, TRIAL_DATASET)

    def test_subjects_are_labelled_as_the_statistics_tables_expect(self):
        """The `subject` column every existing figure groups on."""
        self.assertEqual(self.resolved.label_subject('01'), 'Subject01')


class TestEveryDatasetResolves(unittest.TestCase):
    def test_each_registered_dataset_gives_a_usable_spec(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                spec = tracking_spec(name)
                self.assertTrue(spec.joints, "no joints to track")
                self.assertTrue(spec.sensors, "no sensors named")
                self.assertEqual(len(spec.gravity), 3)
                self.assertGreater(float(np.linalg.norm(spec.gravity)), 9.0)

    def test_only_the_primary_joints_are_benchmarked(self):
        """IMoVE's joint table carries High and Low placement variants as extra pairs. They
        triple the filter runtime and answer a different question, so the tracking spec takes
        the anatomical joints only."""
        spec = get_dataset('imove')
        self.assertGreater(len(spec.joints), len(spec.primary_joints))
        self.assertEqual(tuple(tracking_spec('imove').joints), spec.primary_joints)

    def test_a_spec_names_only_sensors_its_dataset_has(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                declared = set(get_dataset(name).segment_sensor.values())
                self.assertTrue(set(tracking_spec(name).sensors) <= declared)

    def test_an_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            tracking_spec('not_a_dataset')


class TestBiplaneWorldFrame(unittest.TestCase):
    """The biplane half's mocap world is Z-UP, unlike either marker-referenced dataset.

    Pinned because it is the one constant here whose error produces no symptom: the three
    relative-filter arms never consult gravity, so a run would look entirely normal while the
    EKF arm tracked against a reference rotated 90 degrees.
    """

    def test_both_biplane_specs_are_z_up(self):
        for name in ('imove_biplane', 'imove_biplane_vicon'):
            with self.subTest(dataset=name):
                gravity = tracking_spec(name).gravity
                self.assertEqual(int(np.argmax(np.abs(gravity))), 2)
                self.assertGreater(float(gravity[2]), 0.0,
                                   "specific-force convention: a still sensor reads +g")

    def test_the_marker_referenced_datasets_are_y_up(self):
        for name in ('alborno', 'imove'):
            with self.subTest(dataset=name):
                self.assertEqual(int(np.argmax(np.abs(tracking_spec(name).gravity))), 1)

    def test_the_ground_plate_carries_the_spec_gravity(self):
        """Where a wrong world frame actually enters the EKF."""
        spec = tracking_spec('imove_biplane')
        ground = _setup_ekf_ground_plate_([make_plate('lateral_thigh_left__biplane')], spec=spec)
        np.testing.assert_allclose(ground.imu_trace.acc[0], spec.gravity, atol=1e-12)


class TestTwoSpecsOverOneBuild(unittest.TestCase):
    """The biplane build carries every IMU twice, against two ground truths. The pair is the
    measurement — what the reference choice costs — so they must read one build and write two
    namespaces."""

    def setUp(self):
        self.bone = tracking_spec('imove_biplane')
        self.vicon = tracking_spec('imove_biplane_vicon')

    def test_they_read_the_same_build_tree(self):
        self.assertEqual(self.bone.build_dataset, self.vicon.build_dataset)

    def test_they_write_different_output_namespaces(self):
        self.assertNotEqual(self.bone.dataset, self.vicon.dataset)
        self.assertNotEqual(
            paths.joint_angles_path(self.bone.dataset, '12', 'Test1/A/RSDrop1', 'mag_off'),
            paths.joint_angles_path(self.vicon.dataset, '12', 'Test1/A/RSDrop1', 'mag_off'))

    def test_they_name_disjoint_sensors(self):
        """Same devices, different reference, so the plate names differ and neither spec can
        pick up the other's plates from a trial that holds both."""
        self.assertFalse(set(self.bone.sensors) & set(self.vicon.sensors))

    def test_a_spec_with_no_trials_dataset_reads_its_own_name(self):
        self.assertEqual(tracking_spec('alborno').build_dataset, 'alborno')


class TestMethodRefusal(unittest.TestCase):
    """A dataset with no magnetometer must refuse the methods that claim one."""

    def setUp(self):
        self.spec = tracking_spec('imove_biplane')
        self.assertFalse(self.spec.has_magnetometer)

    def test_mag_on_and_mag_adapt_are_refused(self):
        for method in ('mag_on', 'mag_adapt', 'mag_adapt_th500.00'):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    self.spec.check_method(method)

    def test_a_mag_oracle_or_a_distortion_scale_is_refused(self):
        for method in ('mag_off_perfect_mag', 'ekf_perfect_mag', 'mag_off_dist0.50'):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    self.spec.check_method(method)

    def test_the_supported_arms_pass(self):
        """'ekf' is allowed even though its magnetometer channel is equally inert: the name
        claims an absolute filter, not a magnetometer, and an acc-only absolute filter is the
        only absolute baseline this dataset can support."""
        for method in ('marker', 'mag_off', 'ekf', 'mag_off_perfect_acc'):
            with self.subTest(method=method):
                self.spec.check_method(method)

    def test_a_dataset_with_a_magnetometer_refuses_nothing(self):
        for name in ('alborno', 'imove'):
            for method in ('mag_on', 'mag_adapt', 'ekf_perfect_mag'):
                with self.subTest(dataset=name, method=method):
                    tracking_spec(name).check_method(method)

    def test_compute_joint_angles_refuses_before_it_runs_anything(self):
        plates = {'lateral_thigh_left__biplane': make_plate('lateral_thigh_left__biplane')}
        with self.assertRaises(ValueError):
            compute_joint_angles(plates, 'mag_on', tracking=self.spec)


class TestNarrowPlates(unittest.TestCase):
    def setUp(self):
        self.spec = tracking_spec('alborno')

    def test_it_keeps_only_the_spec_sensors(self):
        plates = {name: make_plate(name) for name in
                  ('torso_imu', 'pelvis_imu', 'THIGH_R_M', 'lateral_thigh_left__vicon')}
        self.assertEqual(set(self.spec.narrow_plates(plates)), {'torso_imu', 'pelvis_imu'})

    def test_a_missing_sensor_is_absent_rather_than_an_error(self):
        """A biplane trial images one knee and an Al Borno subject can lose a segment to a
        reconstruction failure. Both are partial trials, not broken ones."""
        plates = {'torso_imu': make_plate('torso_imu')}
        self.assertEqual(set(self.spec.narrow_plates(plates)), {'torso_imu'})

    def test_a_trial_with_no_spec_sensor_at_all_raises(self):
        plates = {'THIGH_R_M': make_plate('THIGH_R_M')}
        with self.assertRaises(ValueError):
            compute_joint_angles(plates, 'marker', tracking=self.spec)


class TestExpectedMagField(unittest.TestCase):
    def test_it_reads_the_spec_reference_sensor(self):
        spec = tracking_spec('imove')
        self.assertEqual(spec.mag_reference, 'PELVIS_M')
        np.testing.assert_allclose(
            _compute_expected_mag_field([make_plate('PELVIS_M')], spec=spec),
            WORLD_MAG, atol=1e-9)

    def test_another_sensors_field_does_not_leak_in(self):
        spec = tracking_spec('imove')
        other = make_plate('THIGH_R_M', world_mag=np.array([500.0, 500.0, 500.0]))
        np.testing.assert_allclose(
            _compute_expected_mag_field([make_plate('PELVIS_M'), other], spec=spec),
            WORLD_MAG, atol=1e-9)

    def test_a_dataset_without_a_magnetometer_gets_exact_zeros(self):
        """The one place zeros are the right answer rather than a silent failure: the reader
        has already zeroed every magnetometer channel, so a nonzero reference would be a
        constant residual the EKF's ground plate would chase."""
        spec = tracking_spec('imove_biplane')
        field = _compute_expected_mag_field([make_plate('lateral_thigh_left__biplane')],
                                            spec=spec)
        np.testing.assert_array_equal(field, np.zeros(3))

    def test_a_magnetometer_dataset_with_no_reference_declared_raises(self):
        spec = TrackingSpec(dataset='x', joints={'J': ('a', 'b')}, gravity=EXPECTED_GRAVITY,
                            mag_reference=None, has_magnetometer=True)
        with self.assertRaises(ValueError):
            _compute_expected_mag_field([make_plate('a')], spec=spec)

    def test_an_absent_reference_sensor_is_not_silently_zero(self):
        spec = tracking_spec('alborno')
        with np.errstate(all='ignore'):
            with self.assertRaises((ValueError, IndexError)):
                _compute_expected_mag_field([make_plate('femur_r_imu')], spec=spec)


class TestUnscoredTrials(unittest.TestCase):
    """The benchmark's shortfall report, which has to survive the two subject-label conventions.

    Its first version compared the raw subject id against the statistics tables' labelled one and
    so reported ALL 19 Al Borno trials as contributing nothing, on the line directly below one
    saying 19 were aggregated. A report that cries wolf on a complete run is worse than none,
    because the case it exists for — 2 of 379 biplane trials genuinely missing — then reads as
    the same noise.
    """

    def _summary(self, rows):
        return pd.DataFrame([{'subject': subject, 'trial': trial, 'rmse_rad': 0.1}
                             for subject, trial in rows])

    def test_a_complete_alborno_run_reports_nothing(self):
        spec = tracking_spec('alborno')
        row_keys = [('01', 'walking'), ('01', 'complexTasks'), ('02', 'walking')]
        summary = self._summary([('Subject01', 'walking'), ('Subject01', 'complexTasks'),
                                 ('Subject02', 'walking')])
        self.assertEqual(unscored_trials(summary, row_keys, spec), [])

    def test_it_names_the_trial_that_is_actually_missing(self):
        spec = tracking_spec('alborno')
        row_keys = [('01', 'walking'), ('02', 'walking')]
        summary = self._summary([('Subject01', 'walking')])
        self.assertEqual(unscored_trials(summary, row_keys, spec), [('02', 'walking')])

    def test_it_works_on_a_dataset_whose_label_is_the_raw_id(self):
        spec = tracking_spec('imove')
        row_keys = [('s13', 't1_walking_001'), ('s14', 't1_walking_001')]
        summary = self._summary([('s13', 't1_walking_001')])
        self.assertEqual(unscored_trials(summary, row_keys, spec),
                         [('s14', 't1_walking_001')])

    def test_an_empty_summary_means_everything_is_missing(self):
        spec = tracking_spec('alborno')
        row_keys = [('01', 'walking')]
        self.assertEqual(unscored_trials(pd.DataFrame(), row_keys, spec), row_keys)

    def test_a_summary_without_a_trial_column_means_everything_is_missing(self):
        """A table written before `trial` existed cannot be matched against, so the honest
        answer is that this cannot confirm any trial was scored."""
        spec = tracking_spec('alborno')
        row_keys = [('01', 'walking')]
        summary = pd.DataFrame([{'subject': 'Subject01', 'rmse_rad': 0.1}])
        self.assertEqual(unscored_trials(summary, row_keys, spec), row_keys)


class TestTrialTask(unittest.TestCase):
    """`trial_type` across datasets: the activity, so pooling by activity works everywhere.

    On Al Borno the trial name IS the activity, which is why the trial key and this column
    were one thing while there was one dataset.
    """

    def test_alborno_names_are_already_activities(self):
        for trial in ('walking', 'complexTasks'):
            self.assertEqual(trial_task('alborno', trial), trial)

    def test_imove_strips_the_task_index_and_the_take_number(self):
        cases = {'t1_walking_001': 'walking', 't4_lat_step_001': 'lat_step',
                 't2_treadmill_walking_001': 'treadmill_walking', 't11_sts_001': 'sts'}
        for trial, expected in cases.items():
            with self.subTest(trial=trial):
                self.assertEqual(trial_task('imove', trial), expected)

    def test_biplane_strips_the_side_and_the_repeat(self):
        for dataset in ('imove_biplane', 'imove_biplane_vicon'):
            with self.subTest(dataset=dataset):
                self.assertEqual(trial_task(dataset, 'Test1/A/LSDrop1'), 'SDrop')
                self.assertEqual(trial_task(dataset, 'Test1/C/RrunStance2'), 'runStance')

    def test_an_unparsed_name_falls_back_to_itself(self):
        """An honest answer — it pools with nothing but itself — rather than a raise, which
        would take down a whole run over a naming convention."""
        self.assertEqual(trial_task('imove', 'oddly_named'), 'oddly_named')
        self.assertEqual(trial_task('some_new_dataset', 'trial7'), 'trial7')


if __name__ == '__main__':
    unittest.main()
