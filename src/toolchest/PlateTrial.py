import os
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import torch
from .MagnetometerCalibration import MagnetometerCalibration
from .DoubleSingleJacobianFilter import DoubleSingleJacobianFilter

from typing import Tuple, List, Optional, Dict

IMU_TO_TRC_NAME_MAP = {'pelvis_imu': 'Pelvis_IMU', 'femur_r_imu': 'R.Femur_IMU', 'femur_l_imu': 'L.Femur_IMU',
                       'tibia_r_imu': 'R.Tibia_IMU', 'tibia_l_imu': 'L.Tibia_IMU', 'calcn_r_imu': 'R.Foot_IMU',
                       'calcn_l_imu': 'L.Foot_IMU', 'torso_imu': 'Back_IMU'}


class PlateTrial:
    """
    This class contains a trace of a plate frame over time. Optionally, this can attach an IMUTrace and manipulate it.
    Or, it can generate a synthetic trace by finite differencing the plate frames over time.
    """

    name: str
    imu_trace: IMUTrace
    world_trace: WorldTrace

    # If this is a double cheeseburger trial
    second_imu_trace: Optional[IMUTrace]
    imu_offset: np.ndarray
    second_imu_offset: np.ndarray

    # Where did this PlateTrial come from?
    subject: str = ''
    task: str = ''

    def __init__(self,
                 name: str,
                 imu_trace: IMUTrace,
                 world_trace: WorldTrace,
                 second_imu_trace: Optional[IMUTrace] = None):
        if len(imu_trace) != len(world_trace):
            print(
                f"IMU and World traces must have the same length. {name} has {len(imu_trace)} IMU samples, with a timestamp range of {min(imu_trace.timestamps)}:{max(imu_trace.timestamps)} and {len(world_trace)} World samples with a timestamp range of {min(world_trace.timestamps)}:{max(world_trace.timestamps)}.")
        if second_imu_trace is not None and len(second_imu_trace) != len(imu_trace):
            print(
                f"Second IMU trace must have the same length as the first IMU trace. {name} has {len(imu_trace)} IMU samples, with a timestamp range of {min(imu_trace.timestamps)}:{max(imu_trace.timestamps)} and {len(second_imu_trace)} second IMU samples with a timestamp range of {min(second_imu_trace.timestamps)}:{max(second_imu_trace.timestamps)}.")
        assert len(imu_trace) == len(world_trace), "IMU and World traces must have the same length"
        if max(np.abs(imu_trace.timestamps - world_trace.timestamps)) > 1e-8:
            print(
                f"IMU and World traces must have the same timestamps. {name} has a maximum difference of {max(np.abs(imu_trace.timestamps - world_trace.timestamps))}.")
            print(f"Index of max difference: {np.argmax(np.abs(imu_trace.timestamps - world_trace.timestamps))}")
            print(
                f"IMU timestamp: {imu_trace.timestamps[np.argmax(np.abs(imu_trace.timestamps - world_trace.timestamps))]}")
            print(
                f"World timestamp: {world_trace.timestamps[np.argmax(np.abs(imu_trace.timestamps - world_trace.timestamps))]}")
            print(f"IMU first 10 timestamps: {imu_trace.timestamps[:10]}")
            print(f"World first 10 timestamps: {world_trace.timestamps[:10]}")
            print(f"IMU last 10 timestamps: {imu_trace.timestamps[-10:]}")
            print(f"World last 10 timestamps: {world_trace.timestamps[-10:]}")
        assert max(np.abs(
            imu_trace.timestamps - world_trace.timestamps)) < 1e-8, "IMU and World traces must have the same timestamps"
        assert isinstance(imu_trace, IMUTrace), "imu_trace must be an instance of IMUTrace"
        assert isinstance(world_trace, WorldTrace), "world_trace must be an instance of WorldTrace"
        self.name = name
        self.imu_trace = imu_trace
        self.world_trace = world_trace
        self.second_imu_trace = second_imu_trace
        if second_imu_trace is None:
            self.imu_offset = np.zeros(3)
            self.second_imu_offset = np.zeros(3)
        else:
            self.imu_offset = np.array([0, 0.03, 0])
            self.second_imu_offset = np.array([0, -0.03, 0])
            # self.imu_offset = np.array([0, -0.05, 0])
            # self.second_imu_offset = np.array([0, 0.05, 0])

    def __len__(self):
        return len(self.imu_trace)

    def __getitem__(self, key):
        return PlateTrial(self.name, self.imu_trace[key], self.world_trace[key], self.second_imu_trace[key])

    def align_imu_trace_to_world_trace(self) -> 'PlateTrial':
        # Generate synthetic world trace
        synthetic_imu_trace = self.world_trace.calculate_imu_trace()
        # Call imu_trace.calculate_rotation_offset_from_gyros(synthetic_world_trace)
        R_wt_it = synthetic_imu_trace.calculate_rotation_offset_from_gyros(self.imu_trace)
        error = synthetic_imu_trace.calculate_gyro_angle_error(self.imu_trace)
        print(self.name + ': ' + str(error))
        # Call imu_trace.rotate_rot
        new_plate = PlateTrial(self.name, self.imu_trace.right_rotate(R_wt_it.T), self.world_trace,
                               self.second_imu_trace.right_rotate(
                                   R_wt_it.T) if self.second_imu_trace is not None else None)
        # new_plate.imu_offset = R_wt_it @ self.imu_offset
        # new_plate.second_imu_offset = R_wt_it @ self.second_imu_offset
        return new_plate

    def rotate_to_world_frame(self) -> 'PlateTrial':
        world_imu_trace = self.imu_trace.left_rotate(self.world_trace.rotations)
        world_second_imu_trace = self.second_imu_trace.left_rotate(
            self.world_trace.rotations) if self.second_imu_trace is not None else None
        new_world_trace = WorldTrace(self.world_trace.timestamps, self.world_trace.positions,
                                     [np.eye(3) for _ in range(len(self.world_trace))])
        return PlateTrial(self.name, world_imu_trace, new_world_trace, world_second_imu_trace)

    def estimate_world_magnetic_field(self) -> np.ndarray:
        """
        This function estimates the world magnetic field from the IMU trace. It does this by averaging the magnetic
        field, rotated to the world frame based on the WorldTrace, over the entire trace.
        """
        median_field = np.median([rot @ mag for rot, mag in zip(self.world_trace.rotations, self.imu_trace.mag)],
                                 axis=0)
        if self.second_imu_trace is not None:
            median_field_2 = np.median(
                [rot @ mag for rot, mag in zip(self.world_trace.rotations, self.second_imu_trace.mag)], axis=0)
            median_field = (median_field + median_field_2) / 2
        return median_field

    def estimate_world_gravity(self) -> np.ndarray:
        """
        This function estimates the world gravity from the IMU trace. It does this by averaging the gravity
        field, rotated to the world frame based on the WorldTrace, over the entire trace.
        """
        median_acc = np.median([rot @ acc for rot, acc in zip(self.world_trace.rotations, self.imu_trace.acc)],
                                    axis=0)

        # Set 9.8 for whatever coordinate is largest
        max_index = np.argmax(np.abs(median_acc))
        negative_gravity = np.zeros(3)
        negative_gravity[max_index] = np.sign(median_acc[max_index]) * 9.81
        # Our accelerometer measures negative gravity, so we need to negate the gravity vector
        return -negative_gravity
    def sphere_calibrate_mags(self) -> 'PlateTrial':
        """
        This function calibrates the magnetic field data by subtracting the sphere fit center from the magnetic field.
        If there are two IMUs, it will calibrate both of them using a shared sphere radius, so that the wanted
        distortions are preserved.
        """
        mag_1_center, mag_1_radius = self.imu_trace.sphere_fit_mag()
        if self.second_imu_trace is not None:
            mag_2_center, mag_2_radius = self.second_imu_trace.sphere_fit_mag()
            print(self.name, "Mag 1 center:", mag_1_center, "Mag 2 center:", mag_2_center, "Mag 1 radius:",
                  mag_1_radius, "Mag 2 radius:", mag_2_radius)
            average_radius = (mag_1_radius + mag_2_radius) / 2
            # mag_1_center, _ = self.imu_trace.sphere_fit_mag(override_radius=average_radius)
            # mag_2_center, _ = self.second_imu_trace.sphere_fit_mag(override_radius=average_radius)
            imu_trace = self.imu_trace.add_offset_to_mag(-mag_1_center)
            second_imu_trace = self.second_imu_trace.add_offset_to_mag(-mag_2_center)
            return PlateTrial(self.name, imu_trace, self.world_trace, second_imu_trace)
        else:
            imu_trace = self.imu_trace.add_offset_to_mag(-mag_1_center)
            return PlateTrial(self.name, imu_trace, self.world_trace)

    @staticmethod
    def calibrate_group_plate_mags(plate_trials: List['PlateTrial'],
                                   joints: List[Tuple[int, int]],
                                   weight_joint_norms: float = 1.0,
                                   weight_plate_mags: float = 0.1,
                                   regularize_bias: float = 0.05,
                                   num_sample_timesteps: int = 5000) -> List['PlateTrial']:
        """
        This function attempts to find a bias offset for each magnetometer such that when applied the resulting chain
        of magnetometer readings is as close to consistent with each other as possible, subject to some other
        regularization.
        """
        # Select 500 random timesteps
        random_timesteps = np.random.choice(len(plate_trials[0]), num_sample_timesteps, replace=False)

        # Precompute data for the joint consistency term
        joint_mag_weights: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        joint_precomputed_mag_projections: List[Tuple[np.ndarray, np.ndarray]] = []
        for parent, child in joints:
            parent_joint_center_offset, child_joint_center_offset, error = plate_trials[
                parent].world_trace.get_joint_center(plate_trials[child].world_trace)
            parent_weight_1, parent_weight_2 = plate_trials[parent].calculate_magnetometer_weights(
                parent_joint_center_offset)
            child_weight_1, child_weight_2 = plate_trials[child].calculate_magnetometer_weights(
                child_joint_center_offset)
            joint_mag_weights.append(((parent_weight_1, parent_weight_2), (child_weight_1, child_weight_2)))
            parent_mag_projection = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            child_mag_projection = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            for i, t in enumerate(random_timesteps):
                parent_mag_projection[i] = torch.from_numpy(
                    plate_trials[parent].imu_trace.mag[t] * parent_weight_1 + plate_trials[parent].second_imu_trace.mag[
                        t] * parent_weight_2)
            for i, t in enumerate(random_timesteps):
                child_mag_projection[i] = torch.from_numpy(
                    plate_trials[child].imu_trace.mag[t] * child_weight_1 + plate_trials[child].second_imu_trace.mag[
                        t] * child_weight_2)
            joint_precomputed_mag_projections.append((parent_mag_projection, child_mag_projection))

        # Precompute data for the plate consistency term
        plate_mags: List[Tuple[np.ndarray, np.ndarray]] = []
        for plate_trial in plate_trials:
            mag_1 = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            mag_2 = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            for i, t in enumerate(random_timesteps):
                mag_1[i] = torch.from_numpy(plate_trial.imu_trace.mag[t])
                mag_2[i] = torch.from_numpy(plate_trial.second_imu_trace.mag[t])
            plate_mags.append((mag_1, mag_2))

        def objective_and_grad(x: np.ndarray):
            x_tensor = torch.tensor(x, requires_grad=True)

            # Decompose the x vector into the magnetometer biases
            plate_mag_biases = []
            for i in range(len(plate_trials)):
                mag_1_bias = x_tensor[i * 6:i * 6 + 3]
                mag_2_bias = x_tensor[i * 6 + 3:i * 6 + 6]
                plate_mag_biases.append((mag_1_bias, mag_2_bias))

            # Calculate the objective function, part by part.
            # 1. The joint consistency term
            joint_consistency_loss = torch.tensor(0.0, dtype=torch.float64)
            for i, (parent, child) in enumerate(joints):
                precomputed_parent_mag_projection, precomputed_child_mag_projection = joint_precomputed_mag_projections[
                    i]
                (parent_weight_1, parent_weight_2), (child_weight_1, child_weight_2) = joint_mag_weights[i]
                parent_bias_1, parent_bias_2 = plate_mag_biases[parent]
                child_bias_1, child_bias_2 = plate_mag_biases[child]

                # Fast vector method
                parent_finals = precomputed_parent_mag_projection + (parent_bias_1 * parent_weight_1) + (
                        parent_bias_2 * parent_weight_2)
                child_finals = precomputed_child_mag_projection + (child_bias_1 * child_weight_1) + (
                        child_bias_2 * child_weight_2)
                parent_norms = torch.linalg.vector_norm(parent_finals, dim=1)
                child_norms = torch.linalg.vector_norm(child_finals, dim=1)
                vector_loss = torch.sum((parent_norms - child_norms) ** 2)
                joint_consistency_loss += vector_loss
                # Slow for loop method (uncomment to check correctness)
                # for_loop_loss = torch.tensor(0.0, dtype=torch.float64)
                # for t in range(len(precomputed_parent_mag_projection)):
                #     parent_final = precomputed_parent_mag_projection[t] + (
                #                 parent_bias_1 * parent_weight_1) + (parent_bias_2 * parent_weight_2)
                #     if not torch.allclose(parent_final, parent_finals[t]):
                #         print('Parent final:', parent_final, 'Parent final vector:', parent_finals[t])
                #     child_final = precomputed_child_mag_projection[t] + (
                #                 child_bias_1 * child_weight_1) + (child_bias_2 * child_weight_2)
                #     if not torch.allclose(child_final, child_finals[t]):
                #         print('Child final:', child_final, 'Child final vector:', child_finals[t])
                #     parent_norm = torch.linalg.vector_norm(parent_final)
                #     if not torch.allclose(parent_norm, parent_norms[t]):
                #         print('Parent norm:', parent_norm, 'Parent norm vector:', parent_norms[t])
                #     child_norm = torch.linalg.vector_norm(child_final)
                #     if not torch.allclose(child_norm, child_norms[t]):
                #         print('Child norm:', child_norm, 'Child norm vector:', child_norms[t])
                #     for_loop_loss += (parent_norm - child_norm) ** 2
                # if not torch.allclose(for_loop_loss, vector_loss, rtol=1e-6):
                #     print('For loop loss:', for_loop_loss.item(), 'Vector loss:', vector_loss.item())
                #     assert torch.allclose(for_loop_loss, vector_loss)
            joint_consistency_loss *= weight_joint_norms

            # 2. The plate consistency term
            plate_consistency_loss = torch.tensor(0.0, dtype=torch.float64)
            for i, (mag_1, mag_2) in enumerate(plate_mags):
                mag_1_bias, mag_2_bias = plate_mag_biases[i]
                # Fast vector method
                vector_loss = torch.sum(
                    torch.norm(mag_1 + mag_1_bias - (mag_2 + mag_2_bias), dim=1) ** 2)
                plate_consistency_loss += vector_loss
                # Slow for loop method (uncomment to check correctness)
                # for_loop_loss = torch.tensor(0.0, dtype=torch.float64)
                # for t in range(len(mag_1)):
                #     for_loop_loss += torch.norm(
                #         mag_1[t] + mag_1_bias - (mag_2[t] + mag_2_bias)) ** 2
                # if not torch.allclose(for_loop_loss, vector_loss, rtol=1e-6):
                #     print('For loop loss:', for_loop_loss.item(), 'Vector loss:', vector_loss.item())
                #     assert torch.allclose(for_loop_loss, vector_loss)
            plate_consistency_loss *= weight_plate_mags

            # 3. The regularization term
            regularization_loss = torch.sum(x_tensor ** 2) * regularize_bias

            loss = joint_consistency_loss + plate_consistency_loss + regularization_loss

            print('Loss: ', loss.item(), ' (Joint Consistency Loss:', joint_consistency_loss.item(),
                  'Plate Consistency Loss:', plate_consistency_loss.item(), 'Regularization Loss:',
                  regularization_loss.item(), ')')

            # Compute gradients
            loss.backward()
            grad = x_tensor.grad.numpy()

            return loss.item(), grad

        # Initialize the x vector
        x0 = np.zeros(len(plate_trials) * 6)

        # Run the optimization
        result = minimize(objective_and_grad, x0, jac=True,
                          method='L-BFGS-B')
        # result = minimize(objective, x0, method='L-BFGS-B')

        # Apply the biases to the IMU traces
        new_plate_trials = []
        for i, plate_trial in enumerate(plate_trials):
            mag_1_bias = result.x[i * 6:i * 6 + 3]
            mag_2_bias = result.x[i * 6 + 3:i * 6 + 6]
            new_imu_trace = plate_trial.imu_trace.add_offset_to_mag(mag_1_bias)
            new_second_imu_trace = plate_trial.second_imu_trace.add_offset_to_mag(mag_2_bias)
            new_plate_trials.append(
                PlateTrial(plate_trial.name, new_imu_trace, plate_trial.world_trace, new_second_imu_trace))
        return new_plate_trials

    @staticmethod
    def calibrate_group_plate_mags_with_mocap(plate_trials: List['PlateTrial'],
                                              joints: List[Tuple[int, int]],
                                              weight_joint_norms: float = 1.0,
                                              weight_plate_mags: float = 0.1,
                                              regularize_bias: float = 0.05,
                                              regularize_sensitivity: float = 1.0,
                                              num_sample_timesteps: int = 500) -> List['PlateTrial']:
        """
        This function attempts to find a bias offset for each magnetometer such that when applied the resulting chain
        of magnetometer readings is as close to consistent with each other as possible, subject to some other
        regularization.
        """
        # Select 500 random timesteps
        random_timesteps = np.random.choice(len(plate_trials[0]), num_sample_timesteps, replace=False)

        # Precompute data for the joint consistency term
        joint_mag_weights: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        joint_precomputed_mag_projections: List[Tuple[torch.Tensor, torch.Tensor]] = []
        joint_rotations: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for parent, child in joints:
            parent_joint_center_offset, child_joint_center_offset, error = plate_trials[
                parent].world_trace.get_joint_center(plate_trials[child].world_trace)
            parent_weight_1, parent_weight_2 = plate_trials[parent].calculate_magnetometer_weights(
                parent_joint_center_offset)
            child_weight_1, child_weight_2 = plate_trials[child].calculate_magnetometer_weights(
                child_joint_center_offset)
            joint_mag_weights.append(((parent_weight_1, parent_weight_2), (child_weight_1, child_weight_2)))
            parent_mag_projection = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            child_mag_projection = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            for i, t in enumerate(random_timesteps):
                parent_mag_projection[i] = torch.from_numpy(
                    plate_trials[parent].imu_trace.mag[t] * parent_weight_1 + plate_trials[parent].second_imu_trace.mag[
                        t] * parent_weight_2)
            for i, t in enumerate(random_timesteps):
                child_mag_projection[i] = torch.from_numpy(
                    plate_trials[child].imu_trace.mag[t] * child_weight_1 + plate_trials[child].second_imu_trace.mag[
                        t] * child_weight_2)
            joint_precomputed_mag_projections.append((parent_mag_projection, child_mag_projection))
            parent_rotations = torch.zeros((len(random_timesteps), 3, 3), dtype=torch.float64, requires_grad=False)
            for i, t in enumerate(random_timesteps):
                parent_rotations[i] = torch.from_numpy(plate_trials[parent].world_trace.rotations[t])
            child_rotations = torch.zeros((len(random_timesteps), 3, 3), dtype=torch.float64, requires_grad=False)
            for i, t in enumerate(random_timesteps):
                child_rotations[i] = torch.from_numpy(plate_trials[child].world_trace.rotations[t])
            joint_rotations.append((parent_rotations, child_rotations))

        # Precompute data for the plate consistency term
        plate_mags: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for plate_trial in plate_trials:
            mag_1 = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            mag_2 = torch.zeros((len(random_timesteps), 3), requires_grad=False)
            for i, t in enumerate(random_timesteps):
                mag_1[i] = torch.from_numpy(plate_trial.imu_trace.mag[t])
                mag_2[i] = torch.from_numpy(plate_trial.second_imu_trace.mag[t])
            plate_mags.append((mag_1, mag_2))

        def objective_and_grad(x: np.ndarray):
            x_tensor = torch.tensor(x, requires_grad=True)

            x_bias = x_tensor[:len(plate_trials) * 6]
            x_sensitivity = x_tensor[len(plate_trials) * 6:]

            # Decompose the x vector into the magnetometer biases
            plate_mag_biases = []
            plate_mag_sensitivity = []
            for i in range(len(plate_trials)):
                mag_1_bias = x_bias[i * 6:i * 6 + 3]
                mag_2_bias = x_bias[i * 6 + 3:i * 6 + 6]
                mag_1_sensitivity = x_sensitivity[i * 6:i * 6 + 3]
                mag_2_sensitivity = x_sensitivity[i * 6 + 3:i * 6 + 6]
                plate_mag_biases.append((mag_1_bias, mag_2_bias))
                plate_mag_sensitivity.append((mag_1_sensitivity, mag_2_sensitivity))

            # Calculate the objective function, part by part.
            # 1. The joint consistency term
            joint_consistency_loss = torch.tensor(0.0, dtype=torch.float64)
            for i, (parent, child) in enumerate(joints):
                parent_mag_1, parent_mag_2 = plate_mags[parent]
                child_mag_1, child_mag_2 = plate_mags[child]
                (parent_weight_1, parent_weight_2), (child_weight_1, child_weight_2) = joint_mag_weights[i]
                parent_rotations, child_rotations = joint_rotations[i]
                parent_bias_1, parent_bias_2 = plate_mag_biases[parent]
                parent_sensitivity_1, parent_sensitivity_2 = plate_mag_sensitivity[parent]
                child_bias_1, child_bias_2 = plate_mag_biases[child]
                child_sensitivity_1, child_sensitivity_2 = plate_mag_sensitivity[child]

                # Fast vector method
                parent_locals = (((parent_mag_1 * parent_sensitivity_1) + parent_bias_1) * parent_weight_1) + (
                            ((parent_mag_2 * parent_sensitivity_2) + parent_bias_2) * parent_weight_2)
                child_locals = (((child_mag_1 * child_sensitivity_1) + child_bias_1) * child_weight_1) + (
                            ((child_mag_2 * child_sensitivity_2) + child_bias_2) * child_weight_2)

                # Add an extra dimension to the vectors to make it (N, 3, 1)
                parent_locals_expanded = parent_locals.unsqueeze(-1)  # Shape (N, 3, 1)
                child_locals_expanded = child_locals.unsqueeze(-1)  # Shape (N, 3, 1)

                # Perform batch matrix multiplication
                parent_worlds = torch.bmm(parent_rotations, parent_locals_expanded)  # Shape (N, 3, 1)
                child_worlds = torch.bmm(child_rotations, child_locals_expanded)  # Shape (N, 3, 1)

                # Remove the extra dimension to get back to (N, 3)
                parent_worlds = parent_worlds.squeeze(-1)  # Shape (N, 3)
                child_worlds = child_worlds.squeeze(-1)  # Shape (N, 3)

                diff = parent_worlds - child_worlds
                diff_norms = torch.linalg.vector_norm(diff, dim=1)
                vector_loss = torch.sum(diff_norms ** 2)
                vector_loss /= len(random_timesteps)
                joint_consistency_loss += vector_loss
                # Slow for loop method (uncomment to check correctness)
                # for_loop_loss = torch.tensor(0.0, dtype=torch.float64)
                # for t in range(len(precomputed_parent_mag_projection)):
                #     parent_local = precomputed_parent_mag_projection[t] + (
                #                 parent_bias_1 * parent_weight_1) + (parent_bias_2 * parent_weight_2)
                #     parent_world = parent_rotations[t] @ parent_local
                #     child_local = precomputed_child_mag_projection[t] + (
                #                 child_bias_1 * child_weight_1) + (child_bias_2 * child_weight_2)
                #     child_world = child_rotations[t] @ child_local
                #     diff = parent_world - child_world
                #     for_loop_loss += torch.linalg.vector_norm(diff) ** 2
                # if not torch.allclose(for_loop_loss, vector_loss, rtol=1e-6):
                #     print('For loop loss:', for_loop_loss.item(), 'Vector loss:', vector_loss.item())
                #     assert torch.allclose(for_loop_loss, vector_loss)
            joint_consistency_loss *= weight_joint_norms

            # 2. The plate consistency term
            plate_consistency_loss = torch.tensor(0.0, dtype=torch.float64)
            for i, (mag_1, mag_2) in enumerate(plate_mags):
                mag_1_bias, mag_2_bias = plate_mag_biases[i]
                mag_1_sensitivity, mag_2_sensitivity = plate_mag_sensitivity[i]
                # Fast vector method
                mag_1_corrected = (mag_1 * mag_1_sensitivity) + mag_1_bias
                mag_2_corrected = (mag_2 * mag_2_sensitivity) + mag_2_bias
                vector_loss = torch.sum(
                    torch.norm(mag_1_corrected - mag_2_corrected, dim=1) ** 2)
                vector_loss /= len(random_timesteps)
                plate_consistency_loss += vector_loss
                # Slow for loop method (uncomment to check correctness)
                # for_loop_loss = torch.tensor(0.0, dtype=torch.float64)
                # for t in range(len(mag_1)):
                #     for_loop_loss += torch.norm(
                #         mag_1[t] + mag_1_bias - (mag_2[t] + mag_2_bias)) ** 2
                # if not torch.allclose(for_loop_loss, vector_loss, rtol=1e-6):
                #     print('For loop loss:', for_loop_loss.item(), 'Vector loss:', vector_loss.item())
                #     assert torch.allclose(for_loop_loss, vector_loss)
            plate_consistency_loss *= weight_plate_mags

            # 3. The regularization term
            regularization_loss = torch.sum(x_bias ** 2) * regularize_bias

            # 4. The sensitivity regularization term
            sensitivity_loss = torch.sum((x_sensitivity - 1.0) ** 2) * regularize_sensitivity

            loss = joint_consistency_loss + plate_consistency_loss + regularization_loss + sensitivity_loss

            print('Loss: ', loss.item(), ' (Joint Consistency Loss:', joint_consistency_loss.item(),
                  'Plate Consistency Loss:', plate_consistency_loss.item(), 'Regularization Loss:',
                  regularization_loss.item(), 'Sensitivity Loss:', sensitivity_loss.item(), ')')

            # Compute gradients
            loss.backward()
            grad = x_tensor.grad.numpy()

            return loss.item(), grad

        # Initialize the x vector
        x0 = np.zeros(len(plate_trials) * 12)

        # Run the optimization
        result = minimize(objective_and_grad, x0, jac=True,
                          method='L-BFGS-B')
        # result = minimize(objective, x0, method='L-BFGS-B')

        # Apply the biases to the IMU traces
        new_plate_trials = []
        bias_result = result.x[:len(plate_trials) * 6]
        sensitivity_result = result.x[len(plate_trials) * 6:]
        for i, plate_trial in enumerate(plate_trials):
            mag_1_bias = bias_result[i * 6:i * 6 + 3]
            mag_2_bias = bias_result[i * 6 + 3:i * 6 + 6]
            sensitivity_result_1 = sensitivity_result[i * 6:i * 6 + 3]
            sensitivity_result_2 = sensitivity_result[i * 6 + 3:i * 6 + 6]
            print(plate_trial.name, "Mag 1 bias:", mag_1_bias, "Mag 2 bias:", mag_2_bias, "Mag 1 sensitivity:",
                  sensitivity_result_1, "Mag 2 sensitivity:", sensitivity_result_2)
            new_imu_trace = plate_trial.imu_trace.scale_mags(sensitivity_result_1).add_offset_to_mag(mag_1_bias)
            new_second_imu_trace = plate_trial.second_imu_trace.scale_mags(sensitivity_result_2).add_offset_to_mag(
                mag_2_bias)
            new_plate_trials.append(
                PlateTrial(plate_trial.name, new_imu_trace, plate_trial.world_trace, new_second_imu_trace))
        return new_plate_trials

    def calculate_magnetometer_weights(self, local_offset: np.ndarray) -> Tuple[float, float]:
        """
        This function calculates the weighted sum for the magnetometer readings based on the local offset from the IMUs.
        """
        local_offset_1 = local_offset - self.imu_offset
        imu_1_to_2 = self.second_imu_offset - self.imu_offset
        imu_1_to_2_dist = np.linalg.norm(imu_1_to_2)
        imu_1_to_2 = imu_1_to_2 / imu_1_to_2_dist
        offset_along_1_to_2 = np.dot(local_offset_1, imu_1_to_2)
        percentage_along_1_to_2 = offset_along_1_to_2 / imu_1_to_2_dist
        # To turn this into a clean weighted sum
        # A + percentage*(B - A) = A + percentage*B - percentage*A = (1 - percentage)*A + percentage*B
        return (1 - percentage_along_1_to_2), percentage_along_1_to_2

    def calculate_grad_distance(self, local_offset: np.ndarray) -> float:
        """
        This function calculates the distance along the gradient from mag 1 to mag 2 to project out.
        """
        local_center = (self.imu_offset + self.second_imu_offset) / 2.0
        imu_1_to_2 = self.second_imu_offset - self.imu_offset
        imu_1_to_2 = imu_1_to_2 / np.linalg.norm(imu_1_to_2)
        dist_from_center = local_offset - local_center
        offset_along_1_to_2 = np.dot(dist_from_center, imu_1_to_2)
        return offset_along_1_to_2

    def project_imu_trace(self, local_offset: np.ndarray, skip_acc: bool = False, skip_mag: bool = False,
                          filter_mag: bool = False) -> IMUTrace:
        """
        This function estimates the values for an IMUTrace at a different location relative to the plate.
        """
        if self.second_imu_trace is None:
            return self.imu_trace.project_acc(local_offset)

        # First we find the local_offset relative to each IMU
        local_offset_1 = local_offset - self.imu_offset
        local_offset_2 = local_offset - self.second_imu_offset

        # Now we can project the accelerometers from each imu, and then average the results
        if skip_acc:
            avg_acc = [0.5 * (acc_1 + acc_2) for acc_1, acc_2 in zip(self.imu_trace.acc, self.second_imu_trace.acc)]
        else:
            projected_imu_1 = self.imu_trace.project_acc(local_offset_1, finite_difference_gyro_method='first_order')
            projected_imu_2 = self.second_imu_trace.project_acc(local_offset_2,
                                                                finite_difference_gyro_method='first_order')
            avg_acc = [0.5 * (acc_1 + acc_2) for acc_1, acc_2 in zip(projected_imu_1.acc, projected_imu_2.acc)]

        # We can also average the gyros
        avg_gyro = [0.5 * (gyro_1 + gyro_2) for gyro_1, gyro_2 in zip(self.imu_trace.gyro, self.second_imu_trace.gyro)]

        if skip_mag:
            avg_mag = [0.5 * (mag_1 + mag_2) for mag_1, mag_2 in zip(self.imu_trace.mag, self.second_imu_trace.mag)]
        else:
            if filter_mag:
                rot_filter = DoubleSingleJacobianFilter()
                rot_filter.new_data_weight_jac = 1e-3
                rot_filter.new_data_weight_mag = 1e-3
                rot_filter.decay_unobserved_jac = 1.0 - 1e-1
                dt: float = self.imu_trace.timestamps[1] - self.imu_trace.timestamps[0]
                grad_distance_to_joint = self.calculate_grad_distance(local_offset)
                mag_to_mag_dist = np.linalg.norm(self.second_imu_offset - self.imu_offset)
                avg_mag = []
                for t in range(len(self.imu_trace)):
                    gyro = (self.imu_trace.gyro[t] + self.second_imu_trace.gyro[t]) / 2.0
                    mag_1 = self.imu_trace.mag[t]
                    mag_2 = self.second_imu_trace.mag[t]
                    grad = (mag_2 - mag_1) / mag_to_mag_dist
                    avg = (mag_1 + mag_2) / 2.0

                    rot_filter.update(gyro, dt, avg, grad)
                    avg_filtered = rot_filter.get_mag_estimate()
                    grad_filtered = rot_filter.get_grad_estimate()
                    local_projection_filtered = avg_filtered + grad_filtered * grad_distance_to_joint
                    avg_mag.append(local_projection_filtered)
            else:
                # To work out the magnetometer reading, we need to work out the weighted sum that's appropriate for this
                # distance from the IMUs
                weight_1, weight_2 = self.calculate_magnetometer_weights(local_offset)

                # Now we can average the magnetometer readings
                avg_mag = [(weight_1 * mag_1) + (weight_2 * mag_2) for mag_1, mag_2 in
                           zip(self.imu_trace.mag, self.second_imu_trace.mag)]

        return IMUTrace(self.imu_trace.timestamps, avg_gyro, avg_acc, avg_mag)

    @staticmethod
    def load_trial_from_folder(folder_path: str, align_plate_trials=True) -> List['PlateTrial']:
        # Parse the folder path to get the subject and task
        subject: str = ''
        task: str = ''
        parts = folder_path.split(os.sep)
        for part in parts:
            if 'Subject' in part:
                subject = part
            if 'complexTasks' in part:
                task = part
            if 'walking' in part:
                task = part

        imu_folder_path = os.path.join(folder_path, 'IMU')
        imu_traces = IMUTrace.load_IMUTraces_from_folder(imu_folder_path)

        # Load the world traces
        world_folder_path = os.path.join(folder_path, 'Mocap')
        trc_file_path = \
            [os.path.join(world_folder_path, file) for file in os.listdir(world_folder_path) if '.trc' in file][0]
        world_traces = WorldTrace.load_from_trc_file(trc_file_path)

        # Time sync and create PlateTrial objects
        plate_trials: List[PlateTrial] = []
        for imu_name, imu_trace in imu_traces.items():
            try:
                world_trace = world_traces[IMU_TO_TRC_NAME_MAP[imu_name]]
            except KeyError:
                print(f"IMU {imu_name} not found in TRC file")
                continue

            synced_imu_trace, synced_world_trace = PlateTrial._sync_imu_trace_to_world_trace(imu_trace, world_trace)
            new_plate_trial = PlateTrial(imu_name, synced_imu_trace, synced_world_trace)
            if align_plate_trials:
                new_plate_trial = new_plate_trial.align_imu_trace_to_world_trace()
            new_plate_trial.subject = subject
            new_plate_trial.task = task
            if len(plate_trials) > 0:
                assert len(plate_trials[0]) == len(new_plate_trial), "All PlateTrials must have the same length"
            plate_trials.append(new_plate_trial)

        return plate_trials

    @staticmethod
    def load_cheeseburger_trial_from_folder(folder_path: str, align_plate_trials=True, max_trc_timestamp=-1.0, max_imu_timestamp=-1.0) -> List[
        'PlateTrial']:
        # This will load a single trial from a folder, where the IMU and TRC files are in the same folder
        parent_dir = os.path.abspath(os.path.dirname(folder_path))
        json_mapping_path = os.path.join(parent_dir, 'mapping.json')
        with open(json_mapping_path, 'r') as f:
            mapping: Dict[str, str] = json.load(f)

        calibration_path = os.path.join(parent_dir, 'magnetometer_calibration.json')
        calibration: Dict = {}
        if os.path.exists(calibration_path):
            with open(calibration_path, 'r') as f:
                calibration: Dict = json.load(f)

        # List files in the folder
        source_csvs: Dict[str, List[str]] = {value: ['', ''] for value in mapping.values()}
        for file in os.listdir(folder_path):
            for key in mapping.keys():
                value = mapping[key]
                if key in file:
                    if file.endswith('_lowg.csv'):
                        source_csvs[value][0] = file
                    if file.endswith('_mag.csv'):
                        source_csvs[value][1] = file

        # Check that all files are present
        for key, files in source_csvs.items():
            if '' in files:
                print(f"Missing file for {key}")
                return []

        # Load the IMU traces
        imu_traces: Dict[str, List[Optional[IMUTrace]]] = {}
        for key, [imu_path, mag_path] in source_csvs.items():
            # This is the raw IMU data for the plate
            # Expected headers are: unix_timestamp_microsec,time_s,ax_m/s/s,ay_m/s/s,az_m/s/s,gx_deg/s,gy_deg/s,gz_deg/s
            imu_path = os.path.join(folder_path, imu_path)
            # This is the raw magnetic data for the plate
            # Expected headers are: unix_timestamp_microsec, time_s, mx_microT, my_microT, mz_microT
            mag_path = os.path.join(folder_path, mag_path)
            # Load the IMU trace data
            timestamps: List[float] = []
            gyros: List[np.ndarray] = []
            accs: List[np.ndarray] = []
            # Read the IMU data from the CSV
            with open(imu_path, 'r') as f:
                lines = f.readlines()
                for l, line in enumerate(lines[1:]):
                    parts = line.split(',')
                    timestamp = float(parts[1])
                    if max_imu_timestamp > 0 and timestamp > max_imu_timestamp:
                        print('Skipping loading IMU data after line', l, 'due to timestamp', timestamp, 'exceeding', max_imu_timestamp)
                        break
                    gyro = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                    acc = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    gyros.append(gyro / 180 * np.pi)
                    accs.append(acc)
                    timestamps.append(timestamp)
            # Read the Mag data from the CSV
            # This is sampled at a lower rate than the IMU data, so we need to duplicate the last value
            mag_timestamps: List[float] = []
            mags: List[np.ndarray] = []
            with open(mag_path, 'r') as f:
                lines = f.readlines()
                for l, line in enumerate(lines[1:]):
                    parts = line.split(',')
                    timestamp = float(parts[1])
                    if max_imu_timestamp > 0 and timestamp > max_imu_timestamp:
                        print('Skipping loading IMU data after line', l, 'due to timestamp', timestamp, 'exceeding', max_imu_timestamp)
                        break
                    mag = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    mag_timestamps.append(timestamp)
                    mags.append(mag)
            # Resample the magnetic data to the IMU data using linear interpolation
            mag_timestamps_np = np.array(mag_timestamps)
            mags_np = np.array(mags)

            # Create an interpolator function for each axis of the magnetometer data
            interp_func_x = interp1d(mag_timestamps_np, mags_np[:, 0], kind='linear', fill_value='extrapolate')
            interp_func_y = interp1d(mag_timestamps_np, mags_np[:, 1], kind='linear', fill_value='extrapolate')
            interp_func_z = interp1d(mag_timestamps_np, mags_np[:, 2], kind='linear', fill_value='extrapolate')

            # Interpolate the magnetometer data to match the IMU timestamps
            interp_mags = np.array([interp_func_x(timestamps), interp_func_y(timestamps), interp_func_z(timestamps)]).T

            if key in calibration:
                print('Calibrating magnetometer data for', key)
                calibration_data = calibration[key]
                calibrator = MagnetometerCalibration()
                calibrator.from_dict(calibration_data)
                interp_mags = calibrator.process(interp_mags)

            # Get into "approximate Earth standard magnetic field" magnitudes
            interp_mags /= 60.0

            # Return to a List[np.ndarray]
            mags = interp_mags.tolist()
            for i in range(len(mags)):
                mags[i] = np.array(mags[i])

            assert len(mags) == len(gyros), "Magnetic data must be the same length as the IMU data"
            assert len(accs) == len(gyros), "Accelerometer data must be the same length as the IMU data"
            assert len(timestamps) == len(gyros), "Timestamps must be the same length as the IMU data"

            imu_trace = IMUTrace(np.array(timestamps), gyros, accs, mags)

            # Assuming `key` is in the format "IMU_R_Foot_1"
            imu_name = '_'.join(key.split('_')[:-1])  # Extracts "R_Foot"
            imu_number = int(key.split('_')[-1]) - 1  # Extracts "1" and converts to 0-based index

            if imu_name not in imu_traces:
                imu_traces[imu_name] = [None, None]  # Assuming only two numbers per IMU

            imu_traces[imu_name][imu_number] = imu_trace

            # print('Loaded IMU trace for', imu_name, '(number', imu_number, ') with', len(imu_trace_1), 'samples')

        #Find the trc file in the folder
        trc_file_path = next((os.path.join(folder_path, file) for file in os.listdir(folder_path) if 'trc' in file and 'smoothed' in file), None)
        world_traces = WorldTrace.load_from_trc_file(trc_file_path, max_trc_timestamp=max_trc_timestamp)

        # Time sync and create PlateTrial objects
        plate_timing = []
        for trace_name, world_trace in world_traces.items():
            imu_name = trace_name
            try:
                imu_trace_1 = imu_traces[imu_name][0]
                imu_trace_2 = imu_traces[imu_name][1]
            except KeyError:
                print(f"IMU {imu_name} not found in loaded IMU traces: {list(imu_traces.keys())}")
                continue
            print('Resampling', imu_name, 'to', world_trace.get_sample_frequency())
            imu_trace_1 = imu_trace_1.resample(world_trace.get_sample_frequency())
            imu_trace_2 = imu_trace_2.resample(world_trace.get_sample_frequency())
            imu_traces[imu_name] = [imu_trace_1, imu_trace_2]

            synthetic_imu_trace = world_trace.calculate_imu_trace()

            imu_1_slice, world_slice = PlateTrial._sync_arrays(np.linalg.norm(imu_trace_1.gyro, axis=1),
                                                               np.linalg.norm(synthetic_imu_trace.gyro, axis=1))
            imu_2_slice, world_slice = PlateTrial._sync_arrays(np.linalg.norm(imu_trace_2.gyro, axis=1),
                                                               np.linalg.norm(synthetic_imu_trace.gyro, axis=1))
            print(
                f"Syncing {imu_name} to {trace_name} with IMU 1 slice {imu_1_slice}/{len(imu_trace_1)}, IMU 2 slice {imu_2_slice}/{len(imu_trace_2)}, and World slice {world_slice}/{len(world_trace)}")
            print(f"IMU 1 start timestamp: {imu_trace_1.timestamps[imu_1_slice][0]}")
            print(f"IMU 2 start timestamp: {imu_trace_2.timestamps[imu_2_slice][0]}")

            plate_timing.append((imu_trace_1.timestamps[imu_1_slice][0],
                                 imu_trace_2.timestamps[imu_2_slice][0],
                                 world_trace.timestamps[world_slice][0],
                                 int(world_slice.stop - world_slice.start)))

        # Isolate median start time and end time since all the IMUs are synced
        imu_1_start_time = np.median([start_time for start_time, _, _, _ in plate_timing])
        imu_2_start_time = np.median([start_time for _, start_time, _, _ in plate_timing])
        imu_start_time = np.mean([imu_1_start_time, imu_2_start_time])
        world_start_time = np.median([start_time for _, _, start_time, _ in plate_timing])
        min_length = int(np.median([length for _, _, _, length in plate_timing]))

        plate_trials = []
        for trace_name, world_trace in world_traces.items():
            imu_name = trace_name
            try:
                imu_trace_1 = imu_traces[imu_name][0].resample(world_trace.get_sample_frequency())
                imu_trace_2 = imu_traces[imu_name][1].resample(world_trace.get_sample_frequency())
            except KeyError:
                print(f"IMU {imu_name} not found in loaded IMU traces: {list(imu_traces.keys())}")
                continue

            # Build the slices for each trace by finding the index nearest to the start time
            imu_1_start_index = np.argmin(np.abs(imu_trace_1.timestamps - imu_start_time))
            imu_1_slice = slice(imu_1_start_index, imu_1_start_index + min_length)
            imu_2_start_index = np.argmin(np.abs(imu_trace_2.timestamps - imu_start_time))
            imu_2_slice = slice(imu_2_start_index, imu_2_start_index + min_length)
            world_start_index = np.argmin(np.abs(world_trace.timestamps - world_start_time))
            world_slice = slice(world_start_index, world_start_index + min_length)

            # Sync the traces
            synced_world_trace = world_trace[world_slice].re_zero_timestamps()
            synced_imu_trace_1 = imu_trace_1[imu_1_slice].re_zero_timestamps()
            synced_imu_trace_2 = imu_trace_2[imu_2_slice].re_zero_timestamps()

            synced_imu_trace_1 = synced_imu_trace_1.resample(synced_world_trace.get_sample_frequency())
            synced_imu_trace_2 = synced_imu_trace_2.resample(synced_world_trace.get_sample_frequency())

            new_plate_trial = PlateTrial(imu_name, synced_imu_trace_1, synced_world_trace, synced_imu_trace_2)
            if align_plate_trials:
                new_plate_trial = new_plate_trial.align_imu_trace_to_world_trace()
            new_plate_trial.subject = 'cheeseburger'
            new_plate_trial.task = 'cheeseburger'
            if len(plate_trials) > 0:
                assert len(plate_trials[0]) == len(new_plate_trial), "All PlateTrials must have the same length"
            plate_trials.append(new_plate_trial)

        return plate_trials

    @staticmethod
    def _sync_imu_trace_to_world_trace(imu_trace: IMUTrace, world_trace: WorldTrace) -> Tuple[IMUTrace, WorldTrace]:
        """
        This works out the time alignment of the IMU trace to the world trace. It returns copies of the traces that are
        time synced.
        """
        # TODO: maybe also add the rotation correction?
        if not np.isclose(imu_trace.get_sample_frequency(), world_trace.get_sample_frequency(), rtol=0.1):
            imu_trace = imu_trace.resample(world_trace.get_sample_frequency())

        synthetic_imu_trace = world_trace.calculate_imu_trace()

        imu_slice, world_slice = PlateTrial._sync_arrays(np.linalg.norm(imu_trace.gyro, axis=1),
                                                         np.linalg.norm(synthetic_imu_trace.gyro, axis=1))

        trimmed_imu_trace = imu_trace[imu_slice].re_zero_timestamps()
        trimmed_world_trace = world_trace[world_slice].re_zero_timestamps()

        return trimmed_imu_trace, trimmed_world_trace

    @staticmethod
    def _sync_arrays(array1: np.array, array2: np.array) -> Tuple[slice, slice]:
        """
        This function takes two arrays and time syncs them. It returns the time synced arrays.
        """
        assert isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray), "Arrays must be numpy arrays"
        assert len(array1) > 0 and len(array2) > 0, "Arrays must have at least one element"
        assert array1.ndim == 1 and array2.ndim == 1, "Arrays must be 1D"
        assert array1.dtype == array2.dtype, "Arrays must have the same dtype"

        array1_len = len(array1)
        array2_len = len(array2)
        max_len = max(array1_len, array2_len)

        array1_padded = np.pad(array1, (0, max_len - array1_len), mode='constant')
        array2_padded = np.pad(array2, (0, max_len - array2_len), mode='constant')
        correlation = signal.correlate(array1_padded, array2_padded, mode='full')
        lag = np.argmax(correlation) - (max_len - 1)
        index1 = max(0, lag)
        index2 = max(0, -lag)
        new_length = min(array1_len - index1, array2_len - index2)
        return slice(index1, index1 + new_length), slice(index2, index2 + new_length)

    def plot_3D_magnetic_field(self, other_trial: Optional['PlateTrial'] = None, plot_slice: Optional[slice] = None):
        if plot_slice is None:
            plot_slice = slice(0, len(self.imu_trace.mag), 100)
        locations: List[np.ndarray] = self.world_trace.positions[plot_slice]
        mags: List[np.ndarray] = self.imu_trace.mag[plot_slice]
        rotations: List[np.ndarray] = self.world_trace.rotations[plot_slice]

        # mags = [rot @ mag for rot, mag in zip(rotations, mags)]

        # Extracting x, y, z positions from locations
        x = np.array([pos[0] for pos in locations])
        y = np.array([pos[1] for pos in locations])
        z = np.array([pos[2] for pos in locations])

        # Extracting magnetic field magnitudes
        u = np.array([mag[0] for mag in mags])
        v = np.array([mag[1] for mag in mags])
        w = np.array([mag[2] for mag in mags])

        # Create a quiver plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True, label=self.name, color='b')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Quiver Plot')

        if other_trial is not None:
            mags: List[np.ndarray] = other_trial.imu_trace.mag[plot_slice]
            rotations: List[np.ndarray] = other_trial.world_trace.rotations[plot_slice]
            mags = [rot @ mag for rot, mag in zip(rotations, mags)]
            # Extracting magnetic field magnitudes
            u = np.array([mag[0] for mag in mags])
            v = np.array([mag[1] for mag in mags])
            w = np.array([mag[2] for mag in mags])
            ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True, label=other_trial.name, color='r')
        plt.legend()
        plt.show()
