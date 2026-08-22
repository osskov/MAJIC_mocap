"""
Covers experiments/sensor_placement.py — what moving the IMU along the segment costs.

WHAT THESE TESTS ARE FOR. This experiment is a comparison of cells that differ in exactly one
thing, and every way of breaking it produces a plausible number rather than a crash:

  * a pair grid that quietly drops the mismatched cells, or double-counts the spec's placement
    VARIANT keys on top of the cross product it already built. Either way the tables still fill
    and the "placement spread" is over a different set of placements than the report claims.
  * a projection applied twice, or applied with the wrong sensor's offset. Both leave the filter
    running and produce joint angles of about the right size.
  * a detrending window that spans a mocap gap, averaging two stretches minutes apart into one
    "constant" offset neither of them had — which SHRINKS the reported fast-tracking error, the
    direction that flatters the method.
  * an error convention transposed. The scalar geodesic angle is symmetric, |log(A^T B)| =
    |log(B^T A)|, so a transposition is invisible in the headline metric and shows up only in the
    detrended one — which is exactly the metric that would then be wrong.
  * a position-invariance control computed across a joint instead of within a segment. A thigh
    and a shank genuinely rotate at different rates, so that version reads ~58 deg/s on real data
    and looks like a violated assumption when nothing is wrong.

So the fixtures below have answers known in CLOSED FORM. A synthetic hinge is built with the
joint centres CHOSEN, then extra "placements" are manufactured on the same rigid bodies by
transporting each plate's pose and accelerometer to a known offset. The truth is then not a
reference measurement but arithmetic: the fitted lever arm must come back as the offset that was
injected, the three placements must read one |omega|, and once projected they must all produce
the same joint angle — while unprojected they must not.

Synthetic throughout — no dataset needed.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import experiments.sensor_placement as sp
from experiments.experiment_utils import _run_relative_filter
from experiments.global_assumptions import DatasetSpec
from experiments.sensor_placement import (ARMS, PlacementPair, SINGLE_PLACEMENT, _indent,
                                          _is_reversed, _segment_of, _segment_roles,
                                          _spread_per_cell, compute_trial, detrended_deg,
                                          error_series, has_placement_contrast,
                                          placement_options, placement_pairs, score,
                                          segment_gyro_spread, summarize)
from src.toolchest.building.assembly import shift_world_origin
from src.toolchest.PlateTrial import PlateTrial
from test.fixtures import generate_3dof_plate, generate_random_plate_trial

FS = 100.0
DURATION = 12.0

# A spec shaped like IMoVE's — three placements on each thigh and shank, one sensor on the pelvis
# and the feet — but with only the right leg, so the expected cell counts are small enough to
# write down by hand in the assertions.
SPEC = DatasetSpec(
    name='fake',
    segment_sensor={
        'Pelvis': 'PELVIS_M',
        'Thigh R High': 'THIGH_R_H', 'Thigh R Mid': 'THIGH_R_M', 'Thigh R Low': 'THIGH_R_L',
        'Shank R High': 'SHANK_R_H', 'Shank R Mid': 'SHANK_R_M', 'Shank R Low': 'SHANK_R_L',
        'Foot R': 'FOOT_R_M',
    },
    joints={'R_Hip': ('PELVIS_M', 'THIGH_R_M'),
            'R_Knee': ('THIGH_R_M', 'SHANK_R_M'),
            'R_Ankle': ('SHANK_R_M', 'FOOT_R_M'),
            # A placement VARIANT key of the kind global_assumptions._imove_joints adds. It names
            # cells the cross product already contains, so the grid must ignore it.
            'R_Knee_H': ('THIGH_R_H', 'SHANK_R_H')},
    primary_joints=('R_Hip', 'R_Knee', 'R_Ankle'),
    pelvis_sensor='PELVIS_M',
    foot_sensors=('FOOT_R_M',),
    subject_label='{}',
    field_reference={},
)

SINGLE_SPEC = DatasetSpec(
    name='fake_single',
    segment_sensor={'Pelvis': 'PELVIS_M', 'Thigh R': 'THIGH_R_M', 'Shank R': 'SHANK_R_M'},
    joints={'R_Hip': ('PELVIS_M', 'THIGH_R_M'), 'R_Knee': ('THIGH_R_M', 'SHANK_R_M')},
    primary_joints=('R_Hip', 'R_Knee'),
    pelvis_sensor='PELVIS_M',
    foot_sensors=(),
    subject_label='{}',
    field_reference={},
)


# The thigh's Mid sensor is 200 mm from the knee, the shank's is 150 mm the other way, and the
# manufactured High/Low placements sit 100 mm either side along the segment's long axis.
THIGH_TO_KNEE = np.array([0.0, 0.0, -0.20])
SHANK_TO_KNEE = np.array([0.0, 0.0, 0.15])
UP_THE_SEGMENT = np.array([0.0, 0.0, 0.10])


def placed(plate: PlateTrial, name: str, offset: np.ndarray) -> PlateTrial:
    """A second sensor on the SAME rigid body as `plate`, `offset` away in its own frame.

    Both halves have to move together and this is the only place in the test suite that says how:
    the pose is transported by `shift_world_origin` (the build's own function) and the
    accelerometer by `project_acc`, which is the rigid-body equation the experiment is testing the
    inverse of. The gyroscope and the magnetometer are NOT transported, because neither depends on
    where on the body it is measured — that invariance is the control `segment_gyro_spread`
    checks, and manufacturing it here is what makes the check meaningful on synthetic data.
    """
    moved = PlateTrial(name, plate.imu_trace.project_acc(offset), plate.world_trace)
    return shift_world_origin(moved, offset)


def hinge_trial(seed: int = 0) -> dict:
    """A thigh and a shank joined by a joint, each carrying three placements.

    Joint centres are CHOSEN, not measured, so every lever arm in the resulting grid is known in
    advance: the Mid thigh sits `THIGH_TO_KNEE` from the knee, and a placement `d` further up the
    segment sits that much further away again.

    SPHERICAL rather than a hinge, and that is not cosmetic. A 1-DOF hinge does not determine its
    own centre — every point along the axis satisfies the constraint equally, so the marker fit
    returns a point that has slid an arbitrary distance along it and `test_the_lever_arm_is_what_
    was_injected` would be asserting against an unidentifiable quantity. Three degrees of freedom
    pin the centre uniquely, which is what makes the injected offset the right answer rather than
    one of infinitely many. Real knees are closer to the hinge, and that degeneracy is a genuine
    property of the estimator this experiment leans on — but it belongs to
    experiments/joint_center.py, which measures it, not here.
    """
    np.random.seed(seed)
    thigh = generate_random_plate_trial(duration=DURATION, fs=FS, add_noise=False)
    shank = generate_3dof_plate(thigh, THIGH_TO_KNEE, SHANK_TO_KNEE, add_noise=False)
    plates = {'THIGH_R_M': PlateTrial('THIGH_R_M', thigh.imu_trace, thigh.world_trace),
              'SHANK_R_M': PlateTrial('SHANK_R_M', shank.imu_trace, shank.world_trace)}
    for segment, plate in (('THIGH_R', thigh), ('SHANK_R', shank)):
        plates[f'{segment}_H'] = placed(plate, f'{segment}_H', UP_THE_SEGMENT)
        plates[f'{segment}_L'] = placed(plate, f'{segment}_L', -UP_THE_SEGMENT)
    return plates


class TestPlacementGrid(unittest.TestCase):
    """The cross product, and what counts as a matched pair."""

    def test_options_include_every_sensor_on_the_segment(self):
        options = placement_options(SPEC)
        self.assertEqual([p for p, _ in options['THIGH_R_M']], ['High', 'Mid', 'Low'])
        self.assertEqual([s for _, s in options['THIGH_R_H']],
                         ['THIGH_R_H', 'THIGH_R_M', 'THIGH_R_L'])

    def test_a_lone_sensor_is_its_own_only_option(self):
        self.assertEqual(placement_options(SPEC)['PELVIS_M'],
                         [(SINGLE_PLACEMENT, 'PELVIS_M')])

    def test_grid_is_the_full_cross_product(self):
        pairs = placement_pairs(SPEC)
        counts = {joint: sum(1 for p in pairs if p.joint == joint)
                  for joint in ('R_Hip', 'R_Knee', 'R_Ankle')}
        # 1x3 at the hip, 3x3 at the knee, 3x1 at the ankle.
        self.assertEqual(counts, {'R_Hip': 3, 'R_Knee': 9, 'R_Ankle': 3})

    def test_variant_joint_keys_are_not_double_counted(self):
        # 'R_Knee_H' names a cell the knee's cross product already contains. If the grid iterated
        # spec.joints instead of spec.primary_joints it would appear again under its own name.
        self.assertNotIn('R_Knee_H', {pair.joint for pair in placement_pairs(SPEC)})

    def test_cells_are_unique(self):
        pairs = placement_pairs(SPEC)
        keys = {(p.joint, p.parent_sensor, p.child_sensor) for p in pairs}
        self.assertEqual(len(keys), len(pairs))

    def test_matched_means_same_height_or_no_choice(self):
        by_pair = {(p.joint, p.pair): p for p in placement_pairs(SPEC)}
        self.assertTrue(by_pair[('R_Knee', 'High-High')].matched)
        self.assertFalse(by_pair[('R_Knee', 'High-Low')].matched)
        # The hip's parent has no alternative, so its cells are as matched as that joint can be.
        # Calling them mismatched would put every hip and ankle cell in the mismatched bucket.
        self.assertTrue(by_pair[('R_Hip', f'{SINGLE_PLACEMENT}-High')].matched)

    def test_narrowing_to_the_plates_present(self):
        pairs = placement_pairs(SPEC, plates={'THIGH_R_M': None, 'SHANK_R_M': None,
                                              'SHANK_R_H': None})
        self.assertEqual({p.pair for p in pairs if p.joint == 'R_Knee'}, {'Mid-Mid', 'Mid-High'})

    def test_a_one_sensor_dataset_has_no_contrast(self):
        self.assertFalse(has_placement_contrast(placement_pairs(SINGLE_SPEC)))
        self.assertTrue(has_placement_contrast(placement_pairs(SPEC)))

    def test_segment_of_strips_only_a_placement_suffix(self):
        self.assertEqual(_segment_of('THIGH_R_H'), 'THIGH_R')
        self.assertEqual(_segment_of('PELVIS_M'), 'PELVIS')
        # No placement suffix to strip, and '_vicon' is not one.
        self.assertEqual(_segment_of('lateral_thigh_left__vicon'), 'lateral_thigh_left__vicon')


class TestErrorMetrics(unittest.TestCase):
    """Closed-form answers for the scoring, including the two conventions that hide."""

    def setUp(self):
        np.random.seed(3)
        # Long enough to have samples on both sides of BURN_IN_S, which the steady scalars need.
        self.n = int((sp.BURN_IN_S + 10.0) * FS)
        self.reference = Rotation.random(self.n, random_state=1).as_matrix()

    def test_identical_series_score_zero(self):
        angle, error = error_series(self.reference, self.reference)
        np.testing.assert_allclose(angle, 0.0, atol=1e-9)
        np.testing.assert_allclose(error, np.eye(3)[None].repeat(self.n, 0), atol=1e-9)

    def test_a_constant_offset_is_reported_as_its_own_angle(self):
        offset = Rotation.from_rotvec(np.radians(7.0) * np.array([0.0, 0.0, 1.0]))
        estimate = np.einsum('tij,jk->tik', self.reference, offset.as_matrix())
        angle, _ = error_series(estimate, self.reference)
        np.testing.assert_allclose(angle, 7.0, atol=1e-9)

    def test_the_error_rotation_is_reference_transpose_times_estimate(self):
        # The scalar angle cannot catch a transposition — it is symmetric — so the matrix itself
        # is asserted. The detrending consumes this matrix, so getting it backwards would silently
        # detrend the inverse error.
        estimate = Rotation.random(self.n, random_state=2).as_matrix()
        _, error = error_series(estimate, self.reference)
        expected = np.einsum('tji,tjk->tik', self.reference, estimate)
        np.testing.assert_allclose(error, expected, atol=1e-12)

    def test_detrending_removes_a_constant_offset_exactly(self):
        offset = Rotation.from_rotvec(np.radians(11.0) * np.array([0.0, 1.0, 0.0]))
        estimate = np.einsum('tij,jk->tik', self.reference, offset.as_matrix())
        _, error = error_series(estimate, self.reference)
        residual = detrended_deg(error, np.arange(self.n), FS)
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_detrending_keeps_within_window_variation(self):
        # A 2 deg wobble inside every window survives; the 11 deg constant around it does not.
        wobble = np.radians(2.0) * np.sin(2 * np.pi * 1.5 * np.arange(self.n) / FS)
        rotvecs = np.zeros((self.n, 3))
        rotvecs[:, 1] = np.radians(11.0) + wobble
        estimate = np.einsum('tij,tjk->tik', self.reference,
                             Rotation.from_rotvec(rotvecs).as_matrix())
        _, error = error_series(estimate, self.reference)
        residual = detrended_deg(error, np.arange(self.n), FS)
        self.assertGreater(residual.mean(), 0.5)
        self.assertLess(residual.mean(), 2.5)

    def test_a_window_never_spans_a_gap(self):
        """Two stretches with different constant offsets, separated by an excluded gap.

        Detrended run by run, each stretch's own offset is removed and the residual is zero. If
        the windows were laid out on wall-clock time rather than on position within the scored
        index, one window would straddle the gap, remove the average of two offsets, and report a
        large fast-tracking error that never happened.
        """
        rotvecs = np.zeros((self.n, 3))
        rotvecs[:200, 2] = np.radians(4.0)
        rotvecs[400:, 2] = np.radians(-9.0)
        estimate = np.einsum('tij,tjk->tik', self.reference,
                             Rotation.from_rotvec(rotvecs).as_matrix())
        _, error = error_series(estimate, self.reference)
        index = np.concatenate([np.arange(0, 200), np.arange(400, self.n)])
        residual = detrended_deg(error, index, FS, window_s=2.0)
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_score_decomposes_a_pure_offset_into_the_slow_channel(self):
        offset = Rotation.from_rotvec(np.radians(6.0) * np.array([1.0, 0.0, 0.0]))
        estimate = np.einsum('tij,jk->tik', self.reference, offset.as_matrix())
        angle, error = error_series(estimate, self.reference)
        timestamps = np.arange(self.n) / FS
        result = score(angle, error, np.arange(self.n), timestamps, FS)
        self.assertAlmostEqual(result['rms_deg'], 6.0, places=6)
        self.assertAlmostEqual(result['detrended_rms_deg'], 0.0, places=4)
        self.assertAlmostEqual(result['offset_deg'], 6.0, places=4)

    def test_steady_scalars_drop_the_head_of_the_record(self):
        angle = np.full(self.n, 2.0)
        angle[:int(sp.BURN_IN_S * FS)] = 40.0
        error = np.eye(3)[None].repeat(self.n, 0)
        timestamps = np.arange(self.n) / FS
        result = score(angle, error, np.arange(self.n), timestamps, FS)
        self.assertGreater(result['rms_deg'], 10.0)
        self.assertAlmostEqual(result['steady_rms_deg'], 2.0, places=6)


class TestRigidBodyInvariance(unittest.TestCase):
    """The controls: what must NOT change when the sensor moves along the segment."""

    @classmethod
    def setUpClass(cls):
        cls.plates = hinge_trial()

    def test_placements_read_one_angular_velocity(self):
        sensors = sp.sensor_rows(self.plates, {}, np.array([0.0, 0.0, 1.0]))
        spread = segment_gyro_spread(sensors)
        self.assertAlmostEqual(spread['THIGH_R'], 0.0, places=9)
        self.assertAlmostEqual(spread['SHANK_R'], 0.0, places=9)

    def test_the_control_is_within_a_segment_not_across_a_joint(self):
        # A thigh and a shank on a moving hinge do NOT share an angular velocity, so a control
        # that compared them would fire on correct data. Asserted so the two can never be
        # confused again.
        sensors = sp.sensor_rows(self.plates, {}, np.array([0.0, 0.0, 1.0]))
        across_the_joint = abs(sensors['THIGH_R_M']['gyro_rms_deg_s']
                               - sensors['SHANK_R_M']['gyro_rms_deg_s'])
        self.assertGreater(across_the_joint, 1.0)

    def test_the_segment_pose_rotation_is_shared(self):
        for a, b in (('THIGH_R_M', 'THIGH_R_H'), ('SHANK_R_M', 'SHANK_R_L')):
            np.testing.assert_allclose(self.plates[a].world_trace.rotations,
                                       self.plates[b].world_trace.rotations, atol=1e-12)

    def test_the_lever_arm_is_what_was_injected(self):
        valid = np.ones(len(self.plates['THIGH_R_M']), dtype=bool)
        expected = {'Mid': THIGH_TO_KNEE, 'High': THIGH_TO_KNEE - UP_THE_SEGMENT,
                    'Low': THIGH_TO_KNEE + UP_THE_SEGMENT}
        for placement, suffix in (('Mid', 'M'), ('High', 'H'), ('Low', 'L')):
            pair = PlacementPair('R_Knee', f'THIGH_R_{suffix}', 'SHANK_R_M', placement, 'Mid')
            offsets = sp.pair_offsets(pair, self.plates, valid, want_inertial=False)
            np.testing.assert_allclose(offsets['parent'], expected[placement], atol=2e-3)
            np.testing.assert_allclose(offsets['child'], SHANK_TO_KNEE, atol=2e-3)


class TestProjection(unittest.TestCase):
    """The correction itself: applied once, with the right offset, and it does its job."""

    @classmethod
    def setUpClass(cls):
        cls.plates = hinge_trial(seed=1)
        cls.valid = np.ones(len(cls.plates['THIGH_R_M']), dtype=bool)

    def test_projecting_here_matches_the_pipelines_own_projected_path(self):
        """`_projected` + project=False must equal `_run_relative_filter(project=True)`.

        The module docstring claims this, and the claim is what justifies doing the transport in
        this file rather than letting the filter do it — the only reason to take it over is that
        the offset has to become an argument so the inertial estimate can be handed in.
        """
        parent, child = self.plates['THIGH_R_M'], self.plates['SHANK_R_M']
        expected = _run_relative_filter(parent, child, project=True, mag_mode='off')
        r_parent, r_child, _ = parent.world_trace.get_joint_center(child.world_trace)
        pair = PlacementPair('R_Knee', 'THIGH_R_M', 'SHANK_R_M', 'Mid', 'Mid')
        actual = sp.run_arm(pair, self.plates, {'parent': r_parent, 'child': r_child},
                            'mocap', 'off')
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_the_projection_collapses_the_placement_spread(self):
        """The experiment's central claim, on data whose joint centres are known exactly.

        Unprojected, the three thigh placements disagree because each carries a different
        lever-arm term into its accelerometer. Projected onto the true joint centre they are
        transported to the SAME point, so what is left is the filter's own error and the spread
        must collapse.
        """
        errors = {'none': [], 'mocap': []}
        for placement, suffix in (('High', 'H'), ('Mid', 'M'), ('Low', 'L')):
            pair = PlacementPair('R_Knee', f'THIGH_R_{suffix}', 'SHANK_R_M', placement, 'Mid')
            offsets = sp.pair_offsets(pair, self.plates, self.valid, want_inertial=False)
            parent = self.plates[pair.parent_sensor]
            child = self.plates[pair.child_sensor]
            reference = np.einsum('tji,tjk->tik', parent.world_trace.rotations,
                                  child.world_trace.rotations)
            for projection in errors:
                estimate = sp.run_arm(pair, self.plates, offsets, projection, 'off')
                angle, _ = error_series(estimate, reference)
                errors[projection].append(float(np.sqrt(np.mean(angle ** 2))))
        unprojected_spread = max(errors['none']) - min(errors['none'])
        projected_spread = max(errors['mocap']) - min(errors['mocap'])
        self.assertGreater(unprojected_spread, 1.0)
        self.assertLess(projected_spread, unprojected_spread / 3.0)

    def test_an_unavailable_inertial_fit_drops_the_cell(self):
        pair = PlacementPair('R_Knee', 'THIGH_R_M', 'SHANK_R_M', 'Mid', 'Mid')
        self.assertIsNone(sp.run_arm(pair, self.plates,
                                     {'parent': THIGH_TO_KNEE, 'child': SHANK_TO_KNEE,
                                      'inertial': None}, 'inertial', 'off'))

    def test_an_unknown_projection_raises(self):
        pair = PlacementPair('R_Knee', 'THIGH_R_M', 'SHANK_R_M', 'Mid', 'Mid')
        with self.assertRaises(ValueError):
            sp.run_arm(pair, self.plates, {'parent': THIGH_TO_KNEE, 'child': SHANK_TO_KNEE},
                       'perfect', 'off')


class TestReportHelpers(unittest.TestCase):
    """The aggregation, where the difference between right and wrong is an ordering."""

    def _stats(self) -> pd.DataFrame:
        rows = []
        # Two trials whose placement ordering is OPPOSITE. Pooling first and then taking a spread
        # would cancel them to zero; taking the spread within each trial keeps both.
        for trial, values in (('t1', {'High-High': 2.0, 'Low-Low': 10.0}),
                              ('t2', {'High-High': 10.0, 'Low-Low': 2.0})):
            for pair, value in values.items():
                rows.append({'subject': 's1', 'trial': trial, 'joint': 'R_Knee', 'pair': pair,
                             'projection': 'none', 'mag': 'on', 'rms_deg': value})
        return pd.DataFrame(rows)

    def test_spread_is_taken_within_a_trial_before_pooling(self):
        spread = _spread_per_cell(self._stats(), 'rms_deg')
        self.assertEqual(len(spread), 2)
        np.testing.assert_allclose(spread['spread_deg'].to_numpy(), [8.0, 8.0])
        self.assertEqual(sorted(spread['best_pair']), ['High-High', 'Low-Low'])

    def test_pooling_first_would_have_hidden_it(self):
        # The mistake the function above avoids, asserted so it stays avoided: the cohort mean of
        # each placement is 6.0 either way, so a spread of pooled means is exactly zero.
        pooled = self._stats().groupby('pair')['rms_deg'].mean()
        self.assertAlmostEqual(pooled.max() - pooled.min(), 0.0)

    def test_reversal_needs_both_ends_to_swap(self):
        self.assertTrue(_is_reversed(['High', 'Mid', 'Low'], ['Low', 'Mid', 'High']))
        self.assertFalse(_is_reversed(['High', 'Mid', 'Low'], ['High', 'Low', 'Mid']))
        # A middle-element swap is one noisy comparison, not a reversal.
        self.assertFalse(_is_reversed(['High', 'Mid', 'Low'], ['High', 'Mid', 'Low']))

    def test_segment_roles_pair_a_segment_with_both_its_joints(self):
        rows = []
        for joint, parent, child in (('R_Hip', 'PELVIS_M', 'THIGH_R_H'),
                                     ('R_Hip', 'PELVIS_M', 'THIGH_R_L'),
                                     ('R_Knee', 'THIGH_R_H', 'SHANK_R_M'),
                                     ('R_Knee', 'THIGH_R_L', 'SHANK_R_M')):
            rows.append({'joint': joint, 'parent_sensor': parent, 'child_sensor': child,
                         'parent_placement': 'Only' if 'PELVIS' in parent else parent[-1],
                         'child_placement': child[-1], 'rms_deg': 1.0})
        roles = _segment_roles(pd.DataFrame(rows))
        self.assertEqual([segment for segment, _, _ in roles], ['THIGH_R'])
        _, as_child, as_parent = roles[0]
        self.assertEqual(set(as_child['joint']), {'R_Hip'})
        self.assertEqual(set(as_parent['joint']), {'R_Knee'})
        # Both frames name THIS segment's placement, whichever role it played.
        self.assertEqual(sorted(as_child['placement']), ['H', 'L'])
        self.assertEqual(sorted(as_parent['placement']), ['H', 'L'])

    def test_summary_is_one_row_per_cell_and_metric(self):
        stats = self._stats()
        for metric in sp.METRICS:
            if metric not in stats.columns:
                stats[metric] = 1.0
        summary = summarize('fake', stats)
        self.assertEqual(set(summary['metric']), set(sp.METRICS))
        knee = summary[(summary.metric == 'rms_deg') & (summary.pair == 'High-High')]
        self.assertEqual(len(knee), 1)
        self.assertEqual(int(knee['n_trials'].iloc[0]), 2)

    def test_indent_pads_every_line_including_the_first(self):
        text = _indent(pd.DataFrame({'a': [1, 2]}), pad='..')
        self.assertTrue(all(line.startswith('..') for line in text.split('\n')))


class TestComputeTrial(unittest.TestCase):
    """End to end on the synthetic hinge: shape, coverage and the skip rule."""

    def test_every_cell_gets_every_arm(self):
        plates = hinge_trial(seed=2)
        tables = compute_trial(plates, SPEC, 'fake', 's1', 't1',
                               tables=('placement_stats', 'placement_geometry'))
        stats = tables['placement_stats']
        # Only the knee: this fixture has no pelvis and no foot, so the hip and ankle cells have
        # no plates and are dropped by the narrowing in `placement_pairs`.
        self.assertEqual(set(stats['joint']), {'R_Knee'})
        self.assertEqual(len(stats), 9 * len(ARMS))
        self.assertEqual(len(tables['placement_geometry']), 9)
        self.assertTrue(stats['rms_deg'].notna().all())

    def test_a_trial_with_no_placement_choice_produces_nothing(self):
        plates = hinge_trial(seed=2)
        single = {name: plate for name, plate in plates.items() if name.endswith('_M')}
        self.assertEqual(compute_trial(single, SINGLE_SPEC, 'fake', 's1', 't1'), {})


if __name__ == '__main__':
    unittest.main()
