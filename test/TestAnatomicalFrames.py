"""
Covers experiments/anatomical_frames.py — the rotation that turns a sensor-frame error into a
flexion / adduction / rotation error, and the split `compute_error_stats` builds from it.

Four things here are wrong in ways no figure would show:

  * THE SIGN CONVENTION. Positive is supposed to mean flexion, adduction and internal rotation on
    BOTH sides of the body, which means the left leg's frontal and transverse axes are flipped
    relative to the right's. Get that wrong and the two legs' errors cancel in every signed
    statistic while RMSE looks fine, so a pooled 'Knee' row quietly averages a real bias to zero.
    Tested on a synthetic MIRRORED subject: the same physical motion imposed on a left and a right
    limb must come out with the same sign on all three axes.

  * THE FRAME COMPOSITION ORDER. `align_world_to_imu` post-multiplies (R_built = R_template C), so
    the basis needs C^T A and not C A. Every wrong order is still orthonormal, still has
    determinant +1, and still produces a full plausible table — and on Al Borno every plate is
    mounted ~180 deg flipped, which makes C^2 a sub-degree rotation and puts this beyond the reach
    of the module's own diagnostics. It is checked here against the definition of the built frame
    instead, at an alignment far enough from 180 deg for the two orders to differ.

  * THE PROJECTION DIRECTION. The components are A^T e, not A e. Both are three numbers of the
    right magnitude; only one of them is the error's components along the anatomical axes.

  * WHETHER THE SPLIT IS A SPLIT AT ALL. Because the basis is orthonormal, per-group RMSE must
    satisfy MAG^2 = FE^2 + AA^2 + IE^2 exactly. That identity is what lets the three anatomical
    panels of a figure be read as a decomposition of the magnitude panel, so it is asserted rather
    than assumed.

All synthetic, all hand-checkable, no dataset needed. What is NOT covered here is whether the
landmarks are in the right place, which no unit test can answer — that is what the module's own
`--validate`, `align_check_deg`, `upright_check_deg` and `hinge_angle_deg` are for.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from experiments.anatomical_frames import (AXIS_SIGNS, SegmentFrame, anatomical_axes, basis_frame,
                                           bases_by_joint, frames_for, parent_sensors, segment_of,
                                           to_sensor_frame)
from experiments.experiment_utils import (ANATOMICAL_AXES, BASIS_COLUMNS, TrackingSpec,
                                          compute_error_stats)

# A synthetic standing subject, in a world where x is anterior, y is superior and z is to the
# subject's right, so an anatomical axis can be recognised on sight. Positions in metres.
RIGHT_LEG = {
    'hip': np.array([0.0, 0.90, 0.09]),
    'knee_lateral': np.array([0.0, 0.45, 0.14]),
    'knee_medial': np.array([0.0, 0.45, 0.04]),
    'ankle_lateral': np.array([0.0, 0.07, 0.13]),
    'ankle_medial': np.array([0.0, 0.07, 0.05]),
}
# The same subject's left leg: mirrored through the sagittal plane, i.e. z negated. 'lateral' is
# still lateral, so it is now the more NEGATIVE z of the pair -- which is the whole point.
LEFT_LEG = {name: point * np.array([1.0, 1.0, -1.0]) for name, point in RIGHT_LEG.items()}

FEMUR_R = SegmentFrame(superior=('hip', 'knee'), rightward=('knee_lateral', 'knee_medial'),
                       primary='superior', side='R')
FEMUR_L = SegmentFrame(superior=('hip', 'knee'), rightward=('knee_medial', 'knee_lateral'),
                       primary='superior', side='L')


def leg_points(leg: dict) -> dict:
    return {**leg, 'knee': 0.5 * (leg['knee_lateral'] + leg['knee_medial'])}


class TestFrameConstruction(unittest.TestCase):
    """The basis is a rotation, and it is the rotation the axis names claim."""

    def test_basis_is_a_right_handed_rotation(self):
        for name, frame, leg in (('right', FEMUR_R, RIGHT_LEG), ('left', FEMUR_L, LEFT_LEG)):
            basis, _ = anatomical_axes(frame, leg_points(leg))
            with self.subTest(side=name):
                np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=12)

    def test_flexion_axis_is_mediolateral(self):
        """FE must be the knee axis, i.e. +-z in this synthetic world, on both sides."""
        for name, frame, leg in (('right', FEMUR_R, RIGHT_LEG), ('left', FEMUR_L, LEFT_LEG)):
            basis, _ = anatomical_axes(frame, leg_points(leg))
            with self.subTest(side=name):
                self.assertAlmostEqual(abs(basis[2, 0]), 1.0, places=6)

    def test_rotation_axis_is_the_long_axis(self):
        """IE must be the segment's long axis, i.e. +-y, since the leg here is vertical."""
        for name, frame, leg in (('right', FEMUR_R, RIGHT_LEG), ('left', FEMUR_L, LEFT_LEG)):
            basis, _ = anatomical_axes(frame, leg_points(leg))
            with self.subTest(side=name):
                self.assertAlmostEqual(abs(basis[1, 2]), 1.0, places=6)

    def test_obliquity_is_the_gram_schmidt_correction(self):
        """A mediolateral pair tilted 20 deg out of perpendicular is reported as 20 deg, and the
        basis is still orthonormal because the primary axis is what survives untouched."""
        points = leg_points(RIGHT_LEG)
        tilt = np.radians(20.0)
        points['knee_lateral'] = points['knee_medial'] + 0.1 * np.array(
            [0.0, np.sin(tilt), np.cos(tilt)])
        basis, geometry = anatomical_axes(FEMUR_R, points)
        self.assertAlmostEqual(geometry['lateral_obliquity_deg'], 20.0, places=4)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        # The long axis is primary, so it is untouched by the bad reference direction.
        self.assertAlmostEqual(abs(basis[1, 2]), 1.0, places=6)

    def test_geometry_reports_segment_dimensions(self):
        _, geometry = anatomical_axes(FEMUR_R, leg_points(RIGHT_LEG))
        self.assertAlmostEqual(geometry['superior_mm'], 450.0, places=3)
        self.assertAlmostEqual(geometry['rightward_mm'], 100.0, places=3)

    def test_zero_length_direction_is_refused(self):
        points = leg_points(RIGHT_LEG)
        points['knee_lateral'] = points['knee_medial']
        with self.assertRaises(ValueError):
            anatomical_axes(FEMUR_R, points)

    def test_a_frame_must_name_its_primary_axis(self):
        with self.assertRaises(ValueError):
            SegmentFrame(rightward=('a', 'b'), primary='superior', side='R')
        with self.assertRaises(ValueError):
            SegmentFrame(rightward=('a', 'b'), primary='rightward', side='R')
        with self.assertRaises(ValueError):
            SegmentFrame(rightward=('a', 'b'), primary='sideways', side='R',
                         superior=('c', 'd'))


class TestFrameComposition(unittest.TestCase):
    """A_imu = C^T A_template, where C is what `align_world_to_imu` post-multiplied.

    This is the one step no diagnostic in the module can check on Al Borno: every plate there is
    mounted ~180 deg flipped, so C and C^T differ by under half a degree and both orders produce
    the same numbers. It is checked here instead, against the definition of the built frame.
    """

    def setUp(self):
        # R_built(t) = R_template(t) C is what assembly.align_world_to_imu leaves behind.
        self.template_to_world = Rotation.from_rotvec([0.2, -0.5, 0.9]).as_matrix()
        self.alignment = Rotation.from_rotvec([0.0, np.pi * 0.98, 0.0]).as_matrix()
        self.built_to_world = self.template_to_world @ self.alignment

    def test_an_axis_of_known_world_direction_survives_the_composition(self):
        """Take an axis that points along world up. Expressed in the template frame it is
        R_template^T y; the same physical axis in the SENSOR frame must be R_built^T y."""
        world_up = np.array([0.0, 1.0, 0.0])
        basis_template = np.column_stack([self.template_to_world.T @ world_up,
                                         *np.eye(3)[1:]])
        basis_sensor = to_sensor_frame(basis_template, self.alignment)
        np.testing.assert_allclose(basis_sensor[:, 0], self.built_to_world.T @ world_up, atol=1e-12)

    def test_the_other_order_is_a_different_answer(self):
        """Guards the test above against passing for a trivial reason: with this 176 deg alignment
        the two orders differ by 8 deg, so a regression to `C @ A` would be caught."""
        basis = Rotation.from_rotvec([0.3, 0.1, -0.2]).as_matrix()
        wrong = self.alignment @ basis
        difference = np.degrees(np.linalg.norm(Rotation.from_matrix(
            to_sensor_frame(basis, self.alignment).T @ wrong).as_rotvec()))
        self.assertGreater(difference, 5.0)

    def test_composition_preserves_orthonormality(self):
        basis = to_sensor_frame(Rotation.from_rotvec([1.0, 0.2, 0.3]).as_matrix(), self.alignment)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=12)


class TestSignConvention(unittest.TestCase):
    """Positive is flexion, adduction and internal rotation — on BOTH sides."""

    def imposed(self, side: str, axis_world: np.ndarray) -> np.ndarray:
        """The (FE, AA, IE) components of a small world-frame rotation about `axis_world`.

        The synthetic segment frames coincide with the world frame here (the leg is vertical and
        faces +x), so a world-frame rotation vector IS a segment-frame one and the basis alone
        decides the answer.
        """
        frame, leg = (FEMUR_R, RIGHT_LEG) if side == 'R' else (FEMUR_L, LEFT_LEG)
        basis, _ = anatomical_axes(frame, leg_points(leg))
        return np.radians(5.0) * (axis_world / np.linalg.norm(axis_world)) @ basis

    def test_flexion_is_positive_on_both_sides(self):
        """Knee flexion swings the distal end POSTERIORLY, which turns the segment's superior
        direction (+y) toward anterior (+x): a rotation about -z, the subject's left."""
        for side in ('R', 'L'):
            with self.subTest(side=side):
                components = self.imposed(side, np.array([0.0, 0.0, -1.0]))
                self.assertGreater(components[0], 0.0)
                np.testing.assert_allclose(components[1:], 0.0, atol=1e-12)

    def test_adduction_is_positive_on_both_sides(self):
        """Adduction moves the distal end TOWARD THE MIDLINE, so the superior direction (+y) turns
        away from it: toward -z for the right leg and toward +z for the left."""
        for side, axis in (('R', np.array([-1.0, 0.0, 0.0])), ('L', np.array([1.0, 0.0, 0.0]))):
            with self.subTest(side=side):
                components = self.imposed(side, axis)
                self.assertGreater(components[1], 0.0)
                np.testing.assert_allclose(components[[0, 2]], 0.0, atol=1e-12)

    def test_internal_rotation_is_positive_on_both_sides(self):
        """Internal rotation turns the anterior direction (+x) toward the midline: toward -z on
        the right and toward +z on the left."""
        for side, axis in (('R', np.array([0.0, 1.0, 0.0])), ('L', np.array([0.0, -1.0, 0.0]))):
            with self.subTest(side=side):
                components = self.imposed(side, axis)
                self.assertGreater(components[2], 0.0)
                np.testing.assert_allclose(components[:2], 0.0, atol=1e-12)

    def test_both_sign_maps_are_right_handed(self):
        for side, matrix in AXIS_SIGNS.items():
            with self.subTest(side=side or 'midline'):
                self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)

    def test_a_mirrored_pair_agrees_on_every_axis(self):
        """The same clinical motion on the two legs is the same numbers, which is what makes a
        pooled left+right row legitimate. Without the left-side flips, AA and IE would be negated
        and any signed statistic over both sides would cancel."""
        for axis_r, axis_l in ((np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0])),
                               (np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
                               (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]))):
            np.testing.assert_allclose(self.imposed('R', axis_r), self.imposed('L', axis_l),
                                       atol=1e-12)


class TestErrorProjection(unittest.TestCase):
    """compute_error_stats' anatomical axes: what they are, and that they add up."""

    @staticmethod
    def trial(error_rotvecs: np.ndarray, joint: str = 'R_Knee',
              subject: str = 'Subject01') -> pd.DataFrame:
        marker = Rotation.from_rotvec(np.zeros((len(error_rotvecs), 3)))
        imu = Rotation.from_rotvec(error_rotvecs) * marker
        timestamps = np.arange(len(error_rotvecs)) * 0.01
        frames = []
        for method, rotation in (('marker', marker), ('mag_adapt', imu)):
            rotvec = rotation.as_rotvec()
            frames.append(pd.DataFrame({
                'subject': subject, 'trial': 'walking', 'trial_type': 'walking',
                'joint_name': joint, 'timestamp': timestamps, 'method': method,
                'rx': rotvec[:, 0], 'ry': rotvec[:, 1], 'rz': rotvec[:, 2]}))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def basis_table(basis: np.ndarray, joint: str = 'R_Knee',
                    subject: str = 'Subject01') -> pd.DataFrame:
        return pd.DataFrame([{'subject': subject, 'trial': 'walking', 'joint_name': joint,
                              **dict(zip(BASIS_COLUMNS, basis.reshape(-1)))}])

    def test_error_along_one_anatomical_axis_lands_on_that_axis(self):
        """A basis that permutes the sensor axes must move the error with it: an error purely
        about the sensor's z must read as pure FE when FE is the sensor's z."""
        basis = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)
        stats = compute_error_stats(self.trial(np.tile([0.0, 0.0, 0.1], (20, 1))),
                                   basis=self.basis_table(basis))
        by_axis = stats.set_index('axis')['rmse_rad']
        self.assertAlmostEqual(by_axis['FE'], 0.1, places=9)
        self.assertAlmostEqual(by_axis['AA'], 0.0, places=9)
        self.assertAlmostEqual(by_axis['IE'], 0.0, places=9)
        self.assertAlmostEqual(by_axis['Z'], 0.1, places=9)

    def test_components_are_the_transpose_projection(self):
        """A^T e, not A e. The two differ for any basis that is not symmetric, and both have the
        right magnitude, so only an explicit case separates them."""
        basis = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
        error = np.tile([0.05, -0.02, 0.11], (16, 1))
        stats = compute_error_stats(self.trial(error), basis=self.basis_table(basis))
        by_axis = stats.set_index('axis')['mean_rad']
        expected = basis.T @ error[0]
        for index, axis in enumerate(ANATOMICAL_AXES):
            self.assertAlmostEqual(by_axis[axis], expected[index], places=9)

    def test_magnitude_is_the_quadrature_sum(self):
        """RMSE_MAG^2 = RMSE_FE^2 + RMSE_AA^2 + RMSE_IE^2, exactly, which is what makes the three
        anatomical panels a decomposition of the magnitude panel rather than a separate result."""
        rng = np.random.default_rng(7)
        basis = Rotation.from_rotvec([0.4, 0.1, -0.9]).as_matrix()
        stats = compute_error_stats(self.trial(rng.normal(scale=0.1, size=(400, 3))),
                                   basis=self.basis_table(basis))
        by_axis = stats.set_index('axis')['rmse_rad']
        self.assertAlmostEqual(
            float(np.hypot(np.hypot(by_axis['FE'], by_axis['AA']), by_axis['IE'])),
            float(by_axis['MAG']), places=12)

    def test_a_rotated_basis_leaves_the_magnitude_alone(self):
        """Every metric on 'MAG', 'X', 'Y' and 'Z' must be untouched by supplying a basis: this is
        an ADDITION to the table, not a reinterpretation of it."""
        error = np.tile([0.02, 0.03, -0.04], (25, 1))
        without = compute_error_stats(self.trial(error))
        with_basis = compute_error_stats(
            self.trial(error), basis=self.basis_table(Rotation.from_rotvec([1.0, 0.2, 0.3])
                                                     .as_matrix()))
        shared = with_basis[with_basis['axis'].isin(['MAG', 'X', 'Y', 'Z'])]
        pd.testing.assert_frame_equal(
            without.sort_values('axis').reset_index(drop=True),
            shared.sort_values('axis').reset_index(drop=True))

    def test_a_joint_with_no_basis_gets_no_anatomical_rows(self):
        """A partial basis must leave the unmatched joint out of the anatomical axes entirely
        rather than filling in an identity, which would relabel sensor axes as anatomical ones."""
        error = np.tile([0.0, 0.0, 0.1], (12, 1))
        df = pd.concat([self.trial(error, joint='R_Knee'),
                        self.trial(error, joint='L_Knee')], ignore_index=True)
        stats = compute_error_stats(df, basis=self.basis_table(np.eye(3), joint='R_Knee'))
        covered = set(stats[stats['axis'].isin(ANATOMICAL_AXES)]['joint_name'])
        self.assertEqual(covered, {'R_Knee'})
        self.assertEqual(set(stats[stats['axis'] == 'MAG']['joint_name']), {'R_Knee', 'L_Knee'})

    def test_basis_is_keyed_per_subject(self):
        """Two subjects with the same joint and different mountings must each get their own basis.
        Keyed only by joint, one subject's frame would be applied to the other's error."""
        error = np.tile([0.0, 0.0, 0.1], (12, 1))
        df = pd.concat([self.trial(error, subject='Subject01'),
                        self.trial(error, subject='Subject02')], ignore_index=True)
        # Subject01: FE is the sensor's z. Subject02: FE is the sensor's x.
        basis = pd.concat([
            self.basis_table(np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float),
                             subject='Subject01'),
            self.basis_table(np.eye(3), subject='Subject02')], ignore_index=True)
        stats = compute_error_stats(df, basis=basis)
        fe = stats[stats['axis'] == 'FE'].set_index('subject')['rmse_rad']
        self.assertAlmostEqual(fe['Subject01'], 0.1, places=9)
        self.assertAlmostEqual(fe['Subject02'], 0.0, places=9)

    def test_dropped_invalid_frames_do_not_shift_the_projection(self):
        """Rows are filtered by `valid` before the projection, which leaves gaps in the frame's
        index while the error array stays dense. Pairing each error with its basis by index LABEL
        rather than by position would silently mix two joints' bases together here, and every
        magnitude in the table would still be correct."""
        error_r = np.tile([0.0, 0.0, 0.1], (40, 1))     # pure sensor-z on the right knee
        error_l = np.tile([0.1, 0.0, 0.0], (40, 1))     # pure sensor-x on the left knee
        frames = []
        for joint, error in (('R_Knee', error_r), ('L_Knee', error_l)):
            trial = self.trial(error, joint=joint)
            # Invalidate an interior stripe, so the surviving labels are gappy and unequal
            # between the two joints.
            valid = np.ones(len(trial), dtype=bool)
            valid[3::7] = False
            frames.append(trial.assign(valid=valid))
        df = pd.concat(frames, ignore_index=True)

        # R_Knee: FE is the sensor's z. L_Knee: FE is the sensor's x.
        basis = pd.concat([
            self.basis_table(np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float),
                             joint='R_Knee'),
            self.basis_table(np.eye(3), joint='L_Knee')], ignore_index=True)
        stats = compute_error_stats(df, basis=basis)
        fe = stats[stats['axis'] == 'FE'].set_index('joint_name')['rmse_rad']
        self.assertAlmostEqual(fe['R_Knee'], 0.1, places=9)
        self.assertAlmostEqual(fe['L_Knee'], 0.1, places=9)
        aa = stats[stats['axis'] == 'AA'].set_index('joint_name')['rmse_rad']
        np.testing.assert_allclose(aa.to_numpy(), 0.0, atol=1e-12)

    def test_a_basis_missing_its_columns_is_refused(self):
        with self.assertRaises(ValueError):
            compute_error_stats(self.trial(np.zeros((5, 3))),
                                basis=pd.DataFrame([{'joint_name': 'R_Knee', 'a11': 1.0}]))

    def test_a_basis_sharing_no_key_is_refused(self):
        table = self.basis_table(np.eye(3)).drop(columns=['subject', 'trial', 'joint_name'])
        table['sensor'] = 'femur_r_imu'
        with self.assertRaises(ValueError):
            compute_error_stats(self.trial(np.zeros((5, 3))), basis=table)

    def test_an_empty_basis_is_the_same_as_none(self):
        error = np.tile([0.01, 0.02, 0.03], (8, 1))
        expected = compute_error_stats(self.trial(error))
        actual = compute_error_stats(self.trial(error), basis=pd.DataFrame())
        pd.testing.assert_frame_equal(expected, actual)


class TestSensorToJointResolution(unittest.TestCase):
    """A basis belongs to a SENSOR; an error belongs to a JOINT. The map between them."""

    SPEC = TrackingSpec(
        dataset='alborno',
        joints={'Lumbar': ('pelvis_imu', 'torso_imu'), 'R_Hip': ('pelvis_imu', 'femur_r_imu'),
                'R_Knee': ('femur_r_imu', 'tibia_r_imu')},
        gravity=np.array([0.0, 9.81, 0.0]), mag_reference='torso_imu',
        subject_label='Subject{}')

    def bases(self) -> pd.DataFrame:
        rows = []
        for index, sensor in enumerate(('pelvis_imu', 'femur_r_imu')):
            matrix = Rotation.from_rotvec([0.1 * (index + 1), 0.0, 0.0]).as_matrix()
            rows.append({'dataset': 'alborno', 'subject': '01', 'trial': 'walking',
                         'sensor': sensor, **dict(zip(BASIS_COLUMNS, matrix.reshape(-1)))})
        return pd.DataFrame(rows)

    def test_one_sensor_supplies_every_joint_it_parents(self):
        """The pelvis parents the lumbar AND the hip, so its basis has to appear under both."""
        frame = basis_frame('alborno', spec=self.SPEC, table=self.bases())
        self.assertEqual(sorted(frame['joint_name']), ['Lumbar', 'R_Hip', 'R_Knee'])
        pelvis = frame[frame['joint_name'].isin(['Lumbar', 'R_Hip'])]
        np.testing.assert_allclose(pelvis.iloc[0][list(BASIS_COLUMNS)].to_numpy(dtype=float),
                                   pelvis.iloc[1][list(BASIS_COLUMNS)].to_numpy(dtype=float))

    def test_the_child_side_is_never_consulted(self):
        """tibia_r_imu is a child only, so a basis for it would be unused — and torso_imu, also a
        child only, has none and must not stop the lumbar from getting the pelvis's."""
        frame = basis_frame('alborno', spec=self.SPEC, table=self.bases())
        self.assertIn('Lumbar', set(frame['joint_name']))

    def test_the_subject_column_is_relabelled_for_the_merge(self):
        """The artifact stores '01' and the statistics tables store 'Subject01'. Emitting the raw
        id matches nothing and silently produces no anatomical rows at all."""
        frame = basis_frame('alborno', spec=self.SPEC, table=self.bases())
        self.assertEqual(set(frame['subject']), {'Subject01'})

    def test_narrowing_by_subject_takes_the_raw_id(self):
        frame = basis_frame('alborno', subject='01', spec=self.SPEC, table=self.bases())
        self.assertFalse(frame.empty)
        self.assertTrue(basis_frame('alborno', subject='Subject01', spec=self.SPEC,
                                    table=self.bases()).empty)

    def test_bases_by_joint_returns_matrices_per_joint(self):
        by_joint = bases_by_joint('alborno', '01', 'walking', spec=self.SPEC, table=self.bases())
        self.assertEqual(sorted(by_joint), ['Lumbar', 'R_Hip', 'R_Knee'])
        for joint, matrix in by_joint.items():
            with self.subTest(joint=joint):
                self.assertEqual(matrix.shape, (3, 3))
                np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)

    def test_a_missing_artifact_is_an_empty_frame_with_the_right_columns(self):
        frame = basis_frame('alborno', spec=self.SPEC, table=pd.DataFrame())
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), ['subject', 'trial', 'joint_name', *BASIS_COLUMNS])

    def test_parent_sensors_excludes_children(self):
        self.assertEqual(parent_sensors(self.SPEC), ('pelvis_imu', 'femur_r_imu'))


class TestDatasetCoverage(unittest.TestCase):
    """Which datasets have a frame definition, and the sensor-to-segment map."""

    def test_alborno_covers_every_parent_of_its_joints(self):
        frames, points = frames_for('alborno')
        for segment, frame in frames.items():
            with self.subTest(segment=segment):
                missing = [name for name in frame.points() if name not in points]
                self.assertEqual(missing, [], f"{segment} names points with no marker table")

    def test_imove_covers_every_parent_of_its_joints(self):
        frames, points = frames_for('imove')
        for segment, frame in frames.items():
            with self.subTest(segment=segment):
                self.assertEqual([name for name in frame.points() if name not in points], [])

    def test_the_biplane_halves_are_refused_by_name(self):
        """A silently empty frame table would produce a statistics file whose anatomical axes are
        all absent, which three steps later is indistinguishable from a filter that failed."""
        for dataset in ('imove_biplane', 'imove_biplane_vicon'):
            with self.subTest(dataset=dataset):
                with self.assertRaises(ValueError) as caught:
                    frames_for(dataset)
                self.assertIn('bone frames', str(caught.exception))

    def test_imove_placements_share_a_segment(self):
        """Three sensors sit on each thigh against one marker cluster, so they share the anatomy —
        and each still gets its own basis, because each was fitted its own mounting rotation."""
        for sensor in ('THIGH_R_H', 'THIGH_R_M', 'THIGH_R_L'):
            with self.subTest(sensor=sensor):
                self.assertEqual(segment_of('imove', sensor), 'THIGH_R')

    def test_alborno_sensors_are_their_own_segments(self):
        self.assertEqual(segment_of('alborno', 'femur_r_imu'), 'femur_r_imu')

    def test_every_side_has_a_sign_map(self):
        for frames, _ in (frames_for('alborno'), frames_for('imove')):
            for segment, frame in frames.items():
                with self.subTest(segment=segment):
                    self.assertIn(frame.side, AXIS_SIGNS)


if __name__ == '__main__':
    unittest.main()
