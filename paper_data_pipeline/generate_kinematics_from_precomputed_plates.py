import os
import nimblephysics as nimble
import numpy as np
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import time
from scipy.stats import kstest, probplot, wilcoxon
import numpy as np

mag_methods = {
    "Mag Free": "mag_free",
    "Never Project": "average",
    "Always Project": "project",
    "Cascade": "",
}


def generate_kinematics_report(b3d_path: str, model_path: str, cached_history_path: str, output_path: str, output_csv: str):
    subject = nimble.biomechanics.SubjectOnDisk(os.path.abspath(b3d_path))
    osim = subject.readOpenSimFile(subject.getNumProcessingPasses() - 1,
                                   geometryFolder=os.path.abspath('../data/Geometry') + '/')
    better_joints_osim = nimble.biomechanics.OpenSimParser.parseOsim(os.path.abspath(model_path),
                                                                     geometryFolder=os.path.abspath(
                                                                         '../data/Geometry') + '/')

    print("Loading skeleton history from cache")
    data = np.load(cached_history_path, allow_pickle=True)
    skeleton_history = {}
    for method in mag_methods.keys():
        skeleton_history[method] = data[method]
    skeleton_history["Markers"] = data["Markers"]

    marker_names = list(osim.markersMap.keys())
    imu_plate_names = list(
        set([name.replace("_1", "").replace("_2", "").replace("_3", "").replace("_4", "") for name in marker_names if
             "IMU" in name]))
    print(imu_plate_names)
    body_to_plates: Dict[str, Set[str]] = {}
    for marker in marker_names:
        if "IMU" in marker:
            plate_name = marker.replace("_1", "").replace("_2", "").replace("_3", "").replace("_4", "")
            body_name = osim.markersMap[marker][0].getName()
            if body_name not in body_to_plates:
                body_to_plates[body_name] = set()
            body_to_plates[body_name].add(plate_name)
    print(body_to_plates)

    body_to_plate_orientations: Dict[str, Dict[str, np.ndarray]] = {}
    for body in body_to_plates:
        body_to_plate_orientations[body] = {}
        for plate in body_to_plates[body]:
            marker_1 = plate + "_1"
            assert marker_1 in marker_names
            assert osim.markersMap[marker_1][0].getName() == body
            marker_2 = plate + "_2"
            assert marker_2 in marker_names
            assert osim.markersMap[marker_2][0].getName() == body
            marker_3 = plate + "_3"
            assert marker_3 in marker_names
            assert osim.markersMap[marker_3][0].getName() == body
            marker_4 = plate + "_4"
            assert marker_4 in marker_names
            assert osim.markersMap[marker_4][0].getName() == body

            marker_o = osim.markersMap[marker_3][1]
            marker_x = osim.markersMap[marker_2][1]
            marker_y = osim.markersMap[marker_4][1]
            marker_d = osim.markersMap[marker_1][1]

            assert not np.isnan(marker_o).any(), "NaN in marker_o"
            assert not np.isnan(marker_d).any(), "NaN in marker_d"
            assert not np.isnan(marker_x).any(), "NaN in marker_x"
            assert not np.isnan(marker_y).any(), "NaN in marker_y"

            # Constructing axis and orientation components
            x_axis_1 = marker_d - marker_x
            x_axis_1 = x_axis_1 / np.linalg.norm(x_axis_1)
            assert not np.isnan(x_axis_1).any(), "NaN in x_axis_1"
            x_axis_2 = marker_y - marker_o
            x_axis_2 = x_axis_2 / np.linalg.norm(x_axis_2)
            assert not np.isnan(x_axis_2).any(), "NaN in x_axis_2"
            x_axis = (x_axis_1 + x_axis_2) / 2
            x_axis = x_axis / np.linalg.norm(x_axis)
            assert not np.isnan(x_axis).any(), "NaN in x_axis"

            y_axis_temp_1 = marker_o - marker_x
            y_axis_temp_1 = y_axis_temp_1 / np.linalg.norm(y_axis_temp_1)
            assert not np.isnan(y_axis_temp_1).any(), "NaN in y_axis_temp_1"
            y_axis_temp_2 = marker_y - marker_d
            y_axis_temp_2 = y_axis_temp_2 / np.linalg.norm(y_axis_temp_2)
            assert not np.isnan(y_axis_temp_2).any(), "NaN in y_axis_temp_2"
            y_axis_temp = (y_axis_temp_1 + y_axis_temp_2) / 2
            y_axis_temp = y_axis_temp / np.linalg.norm(y_axis_temp)
            assert not np.isnan(y_axis_temp).any(), "NaN in y_axis_temp"

            z_axis = np.cross(x_axis, y_axis_temp)
            assert not np.isnan(z_axis).any(), "NaN in z_axis"
            z_axis = z_axis / np.linalg.norm(z_axis)
            assert not np.isnan(z_axis).any(), "NaN in z_axis"
            y_axis = np.cross(z_axis, x_axis)
            assert not np.isnan(y_axis).any(), "NaN in y_axis"

            error_y = np.linalg.norm(y_axis - y_axis_temp_1)
            angle_error = np.arccos(np.clip(np.sum(y_axis * y_axis_temp_1), -1, 1)) * 180 / np.pi
            if np.mean(error_y) > 0.015 or np.mean(angle_error) > 1.0:
                print(f"Mean angle error: {np.mean(angle_error)}")
                print(f"Mean norm of y-y_temp: {np.mean(error_y)}")

            # Saving the location of the marker
            R = np.array([x_axis, y_axis, z_axis]).T

            body_to_plate_orientations[body][plate] = R

    joint_segment_dict = {
        'Hip': ('Pelvis', 'Femur'),
        'Knee': ('Femur', 'Shank'),
        'Ankle': ('Shank', 'Foot'),
        'Lumbar_1': ('Pelvis', 'Sternum'),
        'Lumbar_2': ('Pelvis', 'Torso'),
        'Shoulder_1': ('Sternum', 'Upper_Arm'),
        'Shoulder_2': ('Torso', 'Upper_Arm'),
        'Elbow': ('Upper_Arm', 'Lower_Arm')
    }

    found_joints: List[Tuple[str, str, str, str, np.ndarray, np.ndarray]] = []

    for j in range(better_joints_osim.skeleton.getNumJoints()):
        joint = better_joints_osim.skeleton.getJoint(j)
        if joint.getParentBodyNode() is None:
            continue
        parent_body_name = joint.getParentBodyNode().getName()
        child_body_n = joint.getChildBodyNode().getName()

        print(f"Processing joint: {joint.getName()} Parent: {parent_body_name} Child: {child_body_n}")

        if parent_body_name == 'humerus_r':
            # Skip the ulna_r body
            child_body_n = 'radius_r'
        if parent_body_name in body_to_plates and child_body_n in body_to_plates:
            parent_plates = body_to_plates[parent_body_name]
            child_plates = body_to_plates[child_body_n]
            for parent_plate in parent_plates:
                for child_plate in child_plates:
                    for joint_name, (parent_name, child_name) in joint_segment_dict.items():
                        if parent_name in parent_plate and child_name in child_plate:
                            print(
                                f"Skel Joint: {joint.getName()} Plate Joint: {joint_name} Parent Plate: {parent_plate} Child Plate: {child_plate}")
                            parent_orientation = body_to_plate_orientations[parent_body_name][parent_plate]
                            child_orientation = body_to_plate_orientations[child_body_n][child_plate]
                            print(f"Parent Orientation: {parent_orientation}")
                            print(f"Child Orientation: {child_orientation}")
                            found_joints.append((joint_name, joint.getName(), parent_plate, child_plate,
                                                 parent_orientation, child_orientation))

    # Calculate the mean and standard deviation of the error for each method, for each joint
    method_names = [method for method in skeleton_history.keys()]
    output_positions = {
        method: np.zeros((better_joints_osim.skeleton.getNumDofs() + 1, len(skeleton_history["Markers"]))) for method in
        skeleton_history.keys()
    }
    report = {}
    all_joint_angle_errors_by_method = {method: np.array([]) for method in method_names if 'Markers' not in method}
    joint_angle_error_dict = {}
    for current_joint_name, skel_joint, parent_plate, child_plate, parent_orientation, child_orientation in found_joints:
        print(f"Processing joint: {current_joint_name}")
        joint = better_joints_osim.skeleton.getJoint(skel_joint)
        marker_joint_angles = [joint.getNearestPositionToDesiredRotation(
            parent_orientation @ snapshot[current_joint_name] @ child_orientation.T) for snapshot in
            skeleton_history["Markers"]]

        dof_names = []
        for dof in range(joint.getNumDofs()):
            dof_names.append(joint.getDofName(dof))
        for dof, dof_name in enumerate(dof_names):
            dof_index = better_joints_osim.skeleton.getDof(dof_name).getIndexInSkeleton()
            output_positions['Markers'][dof_index, :] = np.array([angle[dof] for angle in marker_joint_angles])

        joint_angle_errors_by_method = {}
        unsigned_joint_angle_errors_by_method = {}
        for method_name in method_names:
            print(f"\tProcessing method: {method_name}")

            method_joint_angles = [joint.getNearestPositionToDesiredRotation(
                parent_orientation @ snapshot[current_joint_name] @ child_orientation.T) for snapshot in
                                   skeleton_history[method_name]]

            if joint.getType() == nimble.dynamics.EulerJoint.getStaticType():
                axis_order = joint.getAxisOrder()
                for t in range(len(method_joint_angles)):
                    method_joint_angles[t] = nimble.math.roundEulerAnglesToNearest(method_joint_angles[t],
                                                                                   marker_joint_angles[t], axis_order)

            for dof, dof_name in enumerate(dof_names):
                dof_index = better_joints_osim.skeleton.getDof(dof_name).getIndexInSkeleton()
                output_positions[method_name][dof_index, :] = np.array([angle[dof] for angle in method_joint_angles])

            if 'Markers' not in method_name:
                joint_angle_errors = [np.abs(marker_joint_angle - method_joint_angle) for
                                      marker_joint_angle, method_joint_angle in
                                      zip(marker_joint_angles, method_joint_angles)]
                flattened_errors = np.array(joint_angle_errors).flatten()
                all_joint_angle_errors_by_method[method_name] = np.concatenate([all_joint_angle_errors_by_method[method_name], flattened_errors])
                unsigned_joint_angle_errors_by_method[method_name] = [marker_joint_angle - method_joint_angle for
                                                                      marker_joint_angle, method_joint_angle in
                                                                      zip(marker_joint_angles, method_joint_angles)]
                joint_angle_errors = joint_angle_errors[5000:48000]
                joint_angle_error_means = np.mean(joint_angle_errors, axis=0)
                joint_angle_error_medians = np.median(joint_angle_errors, axis=0)
                joint_angle_error_stds = np.std(joint_angle_errors, axis=0)
                joint_angle_error_min = np.min(joint_angle_errors, axis=0)
                joint_angle_error_10th_percentile = np.percentile(joint_angle_errors, 10, axis=0)
                joint_angle_error_30th_percentile = np.percentile(joint_angle_errors, 30, axis=0)
                joint_angle_error_70th_percentile = np.percentile(joint_angle_errors, 70, axis=0)
                joint_angle_error_90th_percentile = np.percentile(joint_angle_errors, 90, axis=0)
                joint_angle_error_max = np.max(joint_angle_errors, axis=0)
                joint_angle_errors_by_method[method_name] = joint_angle_errors

                for dof, dof_name in enumerate(dof_names):
                    key_name = current_joint_name + '_' + dof_name
                    if key_name not in report:
                        report[key_name] = {method_name: {} for method_name in method_names}
                    report[key_name][method_name]['mean_degrees'] = joint_angle_error_means[dof] * 180 / np.pi
                    report[key_name][method_name]['median_degrees'] = joint_angle_error_medians[dof] * 180 / np.pi
                    report[key_name][method_name]['std'] = joint_angle_error_stds[dof] * 180 / np.pi
                    report[key_name][method_name]['min_degrees'] = joint_angle_error_min[dof] * 180 / np.pi
                    report[key_name][method_name]['10th_percentile_degrees'] = joint_angle_error_10th_percentile[
                                                                                   dof] * 180 / np.pi
                    report[key_name][method_name]['30th_percentile_degrees'] = joint_angle_error_30th_percentile[
                                                                                   dof] * 180 / np.pi
                    report[key_name][method_name]['70th_percentile_degrees'] = joint_angle_error_70th_percentile[
                                                                                   dof] * 180 / np.pi
                    report[key_name][method_name]['90th_percentile_degrees'] = joint_angle_error_90th_percentile[
                                                                                   dof] * 180 / np.pi
                    report[key_name][method_name]['max_degrees'] = joint_angle_error_max[dof] * 180 / np.pi

        joint_angle_error_dict[current_joint_name] = unsigned_joint_angle_errors_by_method

    report["All Joints"] = {method_name: {} for method_name in method_names}
    for method_name, errors in all_joint_angle_errors_by_method.items():
        report["All Joints"][method_name]['mean_degrees'] = np.mean(errors) * 180 / np.pi
        report["All Joints"][method_name]['median_degrees'] = np.median(errors) * 180 / np.pi
        report["All Joints"][method_name]['std'] = np.std(errors) * 180 / np.pi
        report["All Joints"][method_name]['min_degrees'] = np.min(errors) * 180 / np.pi
        report["All Joints"][method_name]['10th_percentile_degrees'] = np.percentile(errors, 10) * 180 / np.pi
        report["All Joints"][method_name]['30th_percentile_degrees'] = np.percentile(errors, 30) * 180 / np.pi
        report["All Joints"][method_name]['70th_percentile_degrees'] = np.percentile(errors, 70) * 180 / np.pi
        report["All Joints"][method_name]['90th_percentile_degrees'] = np.percentile(errors, 90) * 180 / np.pi
        report["All Joints"][method_name]['max_degrees'] = np.max(errors) * 180 / np.pi

    with open(output_csv, 'w') as f:
        f.write(
            "Joint_DOF,Method,Mean Error (degrees),Median Error (degrees),Std Error (degrees),Min Error (degrees),10th Percentile Error (degrees),30th Percentile Error (degrees),70th Percentile Error (degrees),90th Percentile Error (degrees),Max Error (degrees)\n")
        for method_name in all_joint_angle_errors_by_method.keys():
            for key_name in report:
                f.write(
                    f"{key_name},{method_name},{report[key_name][method_name]['mean_degrees']},{report[key_name][method_name]['median_degrees']},{report[key_name][method_name]['std']},{report[key_name][method_name]['min_degrees']},{report[key_name][method_name]['10th_percentile_degrees']},{report[key_name][method_name]['30th_percentile_degrees']},{report[key_name][method_name]['70th_percentile_degrees']},{report[key_name][method_name]['90th_percentile_degrees']},{report[key_name][method_name]['max_degrees']}\n")

    timestamps = [time * 0.005 for time in range(len(skeleton_history["Markers"]))]
    for method_name in method_names:
        mot_path = os.path.join(output_path, f"joint_positions_{method_name}.mot")
        nimble.biomechanics.OpenSimParser.saveMot(osim.skeleton, mot_path, timestamps, output_positions[method_name])

    # gui = nimble.NimbleGUI()
    # gui.serve(8080)
    #
    # frame = 0
    #
    # while True:
    #     better_joints_osim.skeleton.setPositions(output_positions["Markers"][:, frame])
    #     gui.nativeAPI().renderSkeleton(better_joints_osim.skeleton, prefix='markers', overrideColor=[1, 0, 0, 1])
    #     better_joints_osim.skeleton.setPositions(output_positions["Cascade"][:, frame])
    #     gui.nativeAPI().renderSkeleton(better_joints_osim.skeleton, prefix='cascade')
    #
    #     frame += 1
    #     if frame >= len(skeleton_history["Markers"]):
    #         frame = 0
    #
    #     time.sleep(0.005)

    return joint_angle_error_dict


if __name__ == "__main__":
    subject_number = 2
    generate_kinematics_report(
        b3d_path=f"../data/S{subject_number}_IMU_Data/S{subject_number}.b3d",
        model_path=f"../data/S{subject_number}_IMU_Data/SimplifiedIMUModel.osim",
        cached_history_path=f"../data/S{subject_number}_IMU_Data/Walking/cached_skeleton_history.npz",
        output_path=f"../data/S{subject_number}_IMU_Data/Walking/",
        output_csv=f"../data/Walking_Results/S{subject_number}_Walking_joint_angles.csv"
    )
