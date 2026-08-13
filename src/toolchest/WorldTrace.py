import os
from pathlib import Path
import pandas as pd
from .IMUTrace import IMUTrace
from typing import List, Tuple, Union
import numpy as np
from .finite_difference_utils import central_difference
from .gyro_utils import finite_difference_rotations
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from typing import Dict


class WorldTrace:
    """
    This class contains a trace of a world frame over time. Optionally, this can attach an IMUTrace and manipulate it.
    Or, it can generate a synthetic trace by finite differencing the world frames over time.
    """

    def __init__(self, timestamps: np.ndarray, positions: Union[List[np.ndarray], np.ndarray],
                 rotations: Union[List[np.ndarray], np.ndarray],
                 valid: Union[List[bool], np.ndarray, None] = None):
        """`valid` marks, per frame, whether this pose is trustworthy GROUND TRUTH.

        Defaults to all-True, so every existing caller keeps its current meaning: a trace
        built by hand or by a test fixture is valid throughout. The build package sets it from
        the marker-reconstruction repair (see building.reconstruction.reconstruct_plate),
        which is the only place that knows a pose was interpolated or left corrupt.

        The mask never affects the arrays. Poses stay on a uniform time grid because
        resampling, filtering and every finite-difference angular velocity here assume one;
        `valid` is how a frame is excluded from ERROR STATISTICS without being excluded from
        the signal a filter integrates through.
        """
        self.timestamps = timestamps
        self.positions = np.asarray(positions)
        self.rotations = np.asarray(rotations)
        self.valid = (np.ones(len(timestamps), dtype=bool) if valid is None
                      else np.asarray(valid, dtype=bool))
        if len(self.valid) != len(timestamps):
            raise ValueError(f"valid has length {len(self.valid)} but the trace has "
                             f"{len(timestamps)} frames.")

    def __len__(self):
        """
        Returns the number of samples in the WorldTrace. This allows us to call len(trace) on a WorldTrace instance.
        """
        return len(self.timestamps)

    def __sub__(self, other: 'WorldTrace') -> 'WorldTrace':
        """
        Allows us to subtract two WorldTrace instances. This will subtract the positions and rotations of the two traces.
        """
        if len(self) != len(other):
            raise ValueError(f"WorldTraces must have the same length to subtract them. Got self {len(self)} and other {len(other)}.")
        assert np.array_equal(self.timestamps[0],
                              other.timestamps[0]), "WorldTraces must have the same start time to subtract them."
        # A difference is only trustworthy where BOTH operands are.
        return WorldTrace(self.timestamps, self.positions - other.positions,
                          np.matmul(self.rotations, other.rotations.transpose(0, 2, 1)),
                          valid=self.valid & other.valid)

    def __getitem__(self, key) -> 'WorldTrace':
        """
        Allows us to use the square bracket notation to access the WorldTrace instance. This allows us to slice the
        WorldTrace instance and access ranges of items with `sub_trace = trace[1:4]`. If we pass an integer, we can
        return the corresponding item as a length 1 trace with `sub_trace = trace[2]` and `len(sub_trace) == 1`.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new WorldTrace instance with the sliced items
            return WorldTrace(self.timestamps[key], self.positions[key], self.rotations[key],
                              valid=self.valid[key])
        else:
            # If key is an integer, return the corresponding item as a length 1 trace
            return WorldTrace(np.array([self.timestamps[key]]), self.positions[key:key+1],
                              self.rotations[key:key+1], valid=self.valid[key:key+1])

    def __eq__(self, other):
        """
        Allows us to compare two WorldTrace instances for equality. This will return True if the timestamps, positions, and
        rotations are all _exactly_ equal.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions.shape[0] != other.positions.shape[0] or self.rotations.shape[0] != other.rotations.shape[0]:
            return False
        return (np.array_equal(self.timestamps, other.timestamps) and
                np.array_equal(self.positions, other.positions) and
                np.array_equal(self.rotations, other.rotations))


    def transform(self, rotate: np.ndarray = np.eye(3), translate: np.ndarray = np.zeros(3)) -> 'WorldTrace':
        """
        This function transforms the WorldTrace by rotating and translating the positions and rotations.
        """
        return WorldTrace(self.timestamps,
                          np.einsum('ij,nj->ni', rotate, self.positions) + translate,
                          np.matmul(rotate, self.rotations),
                          valid=self.valid)

    def allclose(self, other, atol=1e-6):
        """
        Allows us to compare two WorldTrace instances for approximate equality. This will return True if the timestamps,
        positions, and rotations are all approximately equal within the specified tolerance.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions.shape[0] != other.positions.shape[0] or self.rotations.shape[0] != other.rotations.shape[0]:
            return False
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                np.allclose(self.positions, other.positions, atol=atol) and
                np.allclose(self.rotations, other.rotations, atol=atol))
    
    def copy(self) -> 'WorldTrace':
        """
        Returns a deep copy of the WorldTrace object.
        """
        return WorldTrace(
            self.timestamps.copy(),
            self.positions.copy(),
            self.rotations.copy(),
            valid=self.valid.copy()
        )
    
    def finite_difference_world_frame_accelerations(self, acc_from_gravity: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        This function computes the acceleration of the world frame by finite differencing the positions.
        """
        # central_difference treats the columns of an (N, 3) array independently, so
        # both differentiations run on all three axes at once.
        velocity = central_difference(self.positions, self.timestamps)
        return central_difference(velocity, self.timestamps) + acc_from_gravity

    def calculate_imu_trace(self,
                            acc_from_gravity: np.ndarray = np.zeros(3),
                            magnetic_field: np.ndarray = np.zeros(3),
                            skip_lin_acc=False) -> IMUTrace:
        """
        This function computes the IMU trace from the world trace by finite differencing the positions and rotations.
        """
        rotations_np = np.array(self.rotations)
        if not skip_lin_acc:
            world_acc = self.finite_difference_world_frame_accelerations(acc_from_gravity)
            world_acc_np = np.array(world_acc)
            local_acc = np.einsum('nji,nj->ni', rotations_np, world_acc_np)
        else:
            local_acc = np.einsum('nji,j->ni', rotations_np, acc_from_gravity)
        assert isinstance(magnetic_field, np.ndarray)
        local_mag = np.einsum('nji,j->ni', rotations_np, magnetic_field)
        local_gyros = finite_difference_rotations(self.rotations, self.timestamps)
        return IMUTrace(self.timestamps, local_gyros, local_acc, local_mag)

    def re_zero_timestamps(self) -> 'WorldTrace':
        """
        Start timestamps at 0
        """
        return WorldTrace(self.timestamps - self.timestamps[0], self.positions, self.rotations,
                          valid=self.valid)
        
    
    def get_sample_frequency(self):
        """
        This function returns the sample frequency of the WorldTrace.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def resample(self, new_frequency: float, time_shift: float = 0.0) -> 'WorldTrace':
        """Band-limited resample to `new_frequency`, optionally shifted by `time_shift`.

        Downsampling anti-aliases first, so mocap content above the new Nyquist is removed
        rather than folded back into the signal band. That matters more here than it sounds:
        the angular velocity and acceleration this trace produces come from differentiating
        marker positions, which amplifies noise as f^2 and puts the loudest content right at
        the top of the band -- the worst thing to alias down.

        `time_shift` moves the output grid in seconds, which is how a FRACTIONAL alignment
        lag is applied. Aligning by whole samples leaves up to half a sample of error, and
        against a 40 Hz IMU that is 12.5 ms -- large enough to matter during a jump landing.
        Resampling has to happen anyway, so the shift is free here and cannot be recovered
        later.

        Rotations go through sign-continuous quaternions rather than matrix entries or
        axis-angle; see resampling.resample_rotations for the measured comparison and for
        why filtering incremental rotations -- the usual tangent-space recipe -- drifts
        without bound here. `valid` is widened
        by the filter's reach before being sampled, because filtering across a dropout
        corrupts its neighbours silently.
        """
        from .resampling import resample_mask, resample_rotations, resample_values

        old_frequency = self.get_sample_frequency()
        duration = self.timestamps[-1] - self.timestamps[0]
        n_new = int(np.floor(duration * new_frequency)) + 1
        new_timestamps = (self.timestamps[0] + time_shift
                          + np.arange(n_new) / new_frequency)

        positions = resample_values(self.positions, self.timestamps, new_timestamps,
                                    source_rate=old_frequency, target_rate=new_frequency)
        rotations = resample_rotations(self.rotations, self.timestamps, new_timestamps,
                                       source_rate=old_frequency, target_rate=new_frequency)
        valid = resample_mask(self.valid, self.timestamps, new_timestamps,
                              source_rate=old_frequency, target_rate=new_frequency)

        return WorldTrace(new_timestamps, positions, rotations, valid=valid)

    def lowpass_filter(self, cutoff_freq: float, order: int):
        """
        This function applies a lowpass filter to the WorldTrace.
        """
        sample_freq = self.get_sample_frequency()
        nyquist_freq = 0.5 * sample_freq
        cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, cutoff, btype='low') # type: ignore
        positions = filtfilt(b, a, self.positions, axis=0)
        angle_axis = Rotation.from_matrix(self.rotations).as_rotvec()
        angle_axis = filtfilt(b, a, angle_axis, axis=0)
        rotations = Rotation.from_rotvec(angle_axis).as_matrix()
        # filtfilt is non-causal and spreads every sample over the filter's support, so an
        # invalid frame contaminates its neighbours. The mask is carried unchanged rather
        # than dilated: widening it would need the filter's effective support, which varies
        # with order and cutoff, and the caller who low-passes a trace with invalid frames
        # in it has a bigger problem than the mask's exact width.
        return WorldTrace(self.timestamps, positions, rotations, valid=self.valid)

    def get_rotation_errors_deg(self, other_trace: 'WorldTrace') -> np.ndarray:
        """
        This function returns a time series list of the rotation errors in degrees between two WorldTrace instances.
        """
        assert len(self) == len(other_trace), "WorldTraces must have the same length to compare them."
        
        errors = Rotation.from_matrix(np.matmul(self.rotations.transpose(0, 2, 1), other_trace.rotations))
        angle_axis = errors.as_rotvec()
        angle_deg = np.linalg.norm(angle_axis, axis=1) * 180.0 / np.pi
        return angle_deg

    def get_joint_center(self, other_world_trace: 'WorldTrace') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Given two world traces, solve for the best fit constant offset from a joint center.
        This is done by minimizing the sum of the squared differences between the two traces. """
        # Parent is other, child is self
        assert isinstance(other_world_trace, WorldTrace), "Must pass a WorldTrace instance to compare."
        assert len(self) == len(other_world_trace), "WorldTraces must have the same length to compare them."

        parent_loc = self.positions
        child_loc = other_world_trace.positions

        r_c_p = parent_loc - child_loc
        r_c_p = r_c_p.flatten()

        R_w_parent = self.rotations.reshape(-1, 3)
        R_w_child = other_world_trace.rotations.reshape(-1, 3)
        R_w = np.hstack((-R_w_parent, R_w_child))

        offsets, res, rank, S = np.linalg.lstsq(R_w, r_c_p, rcond=None)

        parent_offset = offsets[:3]
        child_offset = offsets[3:]
        error = R_w_parent @ parent_offset - R_w_child @ child_offset + r_c_p
        error = error.reshape(-1, 3)
        return parent_offset, child_offset, error
    
    def get_primary_joint_axis(self, other_world_trace: 'WorldTrace') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the primary axis of a hinge joint between two world traces using
        the Mean Axis method, expressed in both the parent's ('self') and
        child's ('other') coordinate frames.

        This method works by finding the axis of rotation for the relative
        orientation at each time step and then determining the mean direction of
        that axis over the entire trial. This is robust to joints where the
        instantaneous axis of rotation may wobble. The mean direction is found
        by computing the principal eigenvector of the covariance matrix of the
        instantaneous axes.

        Args:
            other_world_trace (WorldTrace): The 'child' segment's world trace.
                                          'self' is assumed to be the 'parent'.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - axis_in_self (np.ndarray): The (3,) unit vector for the primary
                                         axis in the `self` (parent) frame.
            - axis_in_other (np.ndarray): The (3,) unit vector for the primary
                                          axis in the `other` (child) frame.
        """
        assert isinstance(other_world_trace, WorldTrace), "Must pass a WorldTrace instance."
        assert len(self) == len(other_world_trace), "WorldTraces must have the same length."

        def _find_mean_axis(relative_rotations: Rotation) -> np.ndarray:
            """
            Finds the mean axis of rotation from a time series of rotations using
            Principal Component Analysis on the instantaneous axes.
            """
            # Get the rotation vectors (axis * angle) for each time step
            rot_vecs = relative_rotations.as_rotvec()  # shape (N, 3)
            
            # Normalize each rotation vector to get the instantaneous axis of rotation
            # Handle cases where the angle is zero to avoid division by zero
            norms = np.linalg.norm(rot_vecs, axis=1)
            non_zero_mask = norms > 1e-8
            
            # If there's no significant rotation anywhere, we can't determine an axis.
            if not np.any(non_zero_mask):
                # Return a default axis, as no motion was detected.
                return np.array([0., 0., 1.])

            axes = np.zeros_like(rot_vecs)
            # Normalize only the non-zero rotation vectors
            axes[non_zero_mask] = rot_vecs[non_zero_mask] / norms[non_zero_mask, np.newaxis]
            
            # To handle the axis ambiguity (v is the same as -v), we ensure all
            # axes point in the same general direction as the first axis.
            first_axis = axes[np.argmax(non_zero_mask)]
            for i in range(len(axes)):
                if np.dot(axes[i], first_axis) < 0:
                    axes[i] *= -1
            
            # The mean axis is the principal component of the distribution of axes,
            # which is the eigenvector of the covariance matrix corresponding to the
            # largest eigenvalue.
            covariance_matrix = np.cov(axes[non_zero_mask].T)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            
            # The mean axis is the eigenvector with the largest eigenvalue
            mean_axis = eigenvectors[:, np.argmax(eigenvalues)].real
            return mean_axis / np.linalg.norm(mean_axis)

        # --- Step 1: Get Relative Rotations from both perspectives ---
        R_wp_stack = Rotation.from_matrix(self.rotations)
        R_wc_stack = Rotation.from_matrix(other_world_trace.rotations)

        # --- Step 2: Calculate axis in the 'self' (parent) frame ---
        # Use R_pc = R_parent.inv() * R_child
        R_pc_stack = R_wp_stack.inv() * R_wc_stack
        axis_in_self = _find_mean_axis(R_pc_stack)

        # --- Step 3: Calculate axis in the 'other' (child) frame ---
        # Use R_cp = R_child.inv() * R_parent
        R_cp_stack = R_wc_stack.inv() * R_wp_stack
        axis_in_other = _find_mean_axis(R_cp_stack)

        return axis_in_self, axis_in_other
    
