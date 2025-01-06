import numpy as np
from typing import List, Tuple, Optional
import torch
from scipy.optimize import minimize, linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

class RawMarkerPlateTrace:
    """
    This class is used to reconstruct a plate's trajectory through space from a set of partially observed points in a
    cloud.
    """
    width: float
    height: float
    positions: List[np.ndarray]
    width_axis_direction: List[np.ndarray]
    height_axis_direction: List[np.ndarray]
    timesteps: List[float]

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.positions = []
        self.width_axis_direction = []
        self.height_axis_direction = []
        self.timesteps = []

    def add_timestep(self, points: List[np.ndarray], time: float, max_marker_jump_velocity: float=3.0, tol=1e-8) -> bool:
        """
        This function takes a cloud of points, and guesses the plate position and orientation based on the provided
        points and the history so far. It then adds this position and orientation to the list of positions and
        orientations.
        """
        if len(self.positions) == 0:
            position, width_axis, height_axis = self.initialize_to_points(points, tol=tol)
            self.positions.append(position)
            self.width_axis_direction.append(width_axis)
            self.height_axis_direction.append(height_axis)
            self.timesteps.append(time)
            return True

        if len(self.positions) > 3:
            velocity_reference_index = max(0, len(self.positions) - 5)
            velocity = (self.positions[-1] - self.positions[velocity_reference_index]) / (self.timesteps[-1] - self.timesteps[velocity_reference_index])
            width_axis_velocity = (self.width_axis_direction[-1] - self.width_axis_direction[velocity_reference_index]) / (self.timesteps[-1] - self.timesteps[velocity_reference_index])
            height_axis_velocity = (self.height_axis_direction[-1] - self.height_axis_direction[velocity_reference_index]) / (self.timesteps[-1] - self.timesteps[velocity_reference_index])

            projected_position = self.positions[-1] + velocity * (time - self.timesteps[-1])
            projected_width_axis = self.width_axis_direction[-1] + width_axis_velocity * (time - self.timesteps[-1])
            projected_height_axis = self.height_axis_direction[-1] + height_axis_velocity * (time - self.timesteps[-1])

            # Ensure axis are orthogonal and normal
            if np.linalg.norm(projected_width_axis) > 0:
                projected_width_axis = projected_width_axis / np.linalg.norm(projected_width_axis)
            if np.linalg.norm(projected_height_axis) > 0:
                projected_height_axis = projected_height_axis / np.linalg.norm(projected_height_axis)
                projected_height_axis = projected_height_axis - np.dot(projected_width_axis, projected_height_axis) * projected_width_axis
            if np.linalg.norm(projected_height_axis) > 0:
                projected_height_axis = projected_height_axis / np.linalg.norm(projected_height_axis)
        else:
            projected_position = self.positions[-1]
            projected_width_axis = self.width_axis_direction[-1]
            projected_height_axis = self.height_axis_direction[-1]

        dt = time - self.timesteps[-1]

        snap_result = self.snap_to_points(points, projected_position, projected_width_axis, projected_height_axis, max_marker_jump_dist=max_marker_jump_velocity * dt, tol=tol)
        if snap_result is None:
            return False
        position, width_axis, height_axis, loss = snap_result
        self.positions.append(position)
        self.width_axis_direction.append(width_axis)
        self.height_axis_direction.append(height_axis)
        self.timesteps.append(time)
        return True

    def clean_up_points(self, raw_points: List[np.ndarray], position: np.ndarray, width_axis: np.ndarray, height_axis: np.ndarray) -> List[np.ndarray]:
        """
        This takes in a raw point cloud and a position, and orientation, and projects the points onto the plate. It
        outputs the points at the corners of the plate, with a preference for points from the input point cloud if any
        are close enough to the corner. If there are no points at the corner, we instead take the virtual corner. This
        method always returns exactly 4 points, in an exactly consistent order.
        """
        points = self.get_point_corners(position, width_axis, height_axis)
        output_points = []
        for point in points:
            closest_point = None
            closest_distance = float('inf')
            for raw_point in raw_points:
                distance = np.linalg.norm(point - raw_point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = raw_point
            if closest_distance < 0.015:
                output_points.append(closest_point)
            else:
                output_points.append(point)
        return output_points

    def get_point_corners(self, position: np.ndarray, width_axis: np.ndarray, height_axis: np.ndarray) -> List[np.ndarray]:
        """
        This function returns the 4 corners of the plate given the position, width axis, and height axis.
        """
        return [position,
                position + (width_axis * self.width),
                position + (height_axis * self.height),
                position + (width_axis * self.width) + (height_axis * self.height)]

    def initialize_to_points(self, points: List[np.ndarray], tol=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This will take an arbitrary cloud of points, and assign them to the closest relevant point on the plate. Then
        it will return the position, width axis, and height axis that best fits the points given their arbitrary
        assignments.
        """
        best_position = None
        best_width_axis = None
        best_height_axis = None
        best_loss = float('inf')

        num_iterations = 100  # Number of random shuffles to try

        for _ in range(num_iterations):
            shuffled_points = points.copy()
            np.random.shuffle(shuffled_points)
            shuffled_points = shuffled_points[:4]
            initial_guess = np.concatenate((shuffled_points[0], shuffled_points[1] - shuffled_points[0], shuffled_points[2] - shuffled_points[0]))

            # Mask indicating which points are available (assume all points are available initially)
            mask = [True, True, True, len(shuffled_points) > 3]

            # If we have 3 points, pad the list with a fourth point
            if len(shuffled_points) == 3:
                shuffled_points.append(np.zeros(3))

            position, width_axis, height_axis, loss = self.fit_point_corners(shuffled_points, initial_guess, mask, tol=tol)

            if loss < best_loss:
                best_loss = loss
                best_position = position
                best_width_axis = width_axis
                best_height_axis = height_axis

        return best_position, best_width_axis, best_height_axis

    def snap_to_points(self, points: List[np.ndarray], position: np.ndarray, width_axis: np.ndarray, height_axis: np.ndarray, max_marker_jump_dist: float=0.01, tol=1e-8) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        This function will take the provided position, width axis, and height axis, and assign each provided point to
        its nearest corner given the current plate configuration. It will then fit the plate to these new points, and
        return the result.
        """
        current_points = self.get_point_corners(position, width_axis, height_axis)
        num_current_points = len(current_points)
        num_points = len(points)

        # Compute the cost matrix
        cost_matrix = np.zeros((num_current_points, num_points))
        for i, c_point in enumerate(current_points):
            for j, point in enumerate(points):
                cost_matrix[i, j] = np.linalg.norm(c_point - point)
                if cost_matrix[i, j] > max_marker_jump_dist:
                    cost_matrix[i, j] = float('inf')

        # Solve the assignment problem using the Hungarian algorithm
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except:
            # If the cost matrix is infeasible, we return None
            return None

        # Create nearest_points and mask based on the optimal assignment
        nearest_points = [np.zeros(3) for _ in range(num_current_points)]
        mask = [False] * num_current_points
        for i, j in zip(row_ind, col_ind):
            nearest_points[i] = points[j]
            mask[i] = True

        # Check if we have accidentally flipped the point assignments
        for i in range(num_current_points):
            if not mask[i]:
                continue
            for j in range(num_current_points):
                if not mask[j] or i == j:
                    continue
                expected_dist = np.linalg.norm(current_points[i] - current_points[j])
                actual_dist = np.linalg.norm(nearest_points[i] - nearest_points[j])
                dist_error = np.abs(expected_dist - actual_dist)
                if dist_error > 0.015:
                    print('Warning: Points are wrong distance apart. Expected:', expected_dist, 'Actual:', actual_dist, 'Error:', dist_error)

        initial_guess = np.concatenate((position, width_axis, height_axis))
        return self.fit_point_corners(nearest_points, initial_guess, mask, tol=tol)


    def fit_point_corners(self, points: List[np.ndarray], initial_guess: np.ndarray, mask: List[bool], tol=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        This function fits the plate's position and orientation to the provided points. It returns the position and
        orientation that best fits the points.
        """
        points_tensor = torch.tensor(np.array(points), dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        def objective(x):
            x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            corners = self.get_concatenated_point_corners(x_tensor)
            masked_corners = corners.view(-1, 3) * mask_tensor.unsqueeze(1)
            masked_points = points_tensor * mask_tensor.unsqueeze(1)
            loss = torch.nn.functional.mse_loss(masked_corners, masked_points, reduction='sum') / mask_tensor.sum()
            loss.backward()
            return loss.item(), x_tensor.grad.numpy().astype(np.float64)

        result = minimize(objective, initial_guess.astype(np.float64), jac=True, method='L-BFGS-B', tol=tol)

        x_optimized = result.x
        loss = result.fun
        position = x_optimized[:3]
        width_axis = x_optimized[3:6]
        height_axis = x_optimized[6:9]

        # Ensure axis are orthogonal and normal
        if np.linalg.norm(width_axis) > 0:
            width_axis = width_axis / np.linalg.norm(width_axis)
        else:
            print('Warning: Width axis is zero length')
            width_axis = np.array([1., 0., 0.])
        if np.linalg.norm(height_axis) > 0:
            height_axis = height_axis / np.linalg.norm(height_axis)
        else:
            print('Warning: Height axis is zero length')
            height_axis = np.array([0., 1., 0.])
        height_axis = height_axis - np.dot(width_axis, height_axis) * width_axis
        if np.linalg.norm(height_axis) > 0:
            height_axis = height_axis / np.linalg.norm(height_axis)
        else:
            print('Warning: Height axis is zero length after orthogonalization')
            height_axis = np.array([0., 1., 0.])

        return position, width_axis, height_axis, loss

    def get_concatenated_point_corners(self, x: torch.Tensor) -> torch.Tensor:
        position = x[:3]
        width_axis = x[3:6]
        height_axis = x[6:9]

        # Ensure axis are orthogonal and normal
        if torch.norm(width_axis) > 0:
            width_axis = width_axis / torch.norm(width_axis)
        else:
            width_axis = torch.tensor([1., 0., 0.])
        if torch.norm(height_axis) > 0:
            height_axis = height_axis / torch.norm(height_axis)
        else:
            height_axis = torch.tensor([0., 1., 0.])
        height_axis = height_axis - torch.dot(width_axis, height_axis) * width_axis
        if torch.norm(height_axis) > 0:
            height_axis = height_axis / torch.norm(height_axis)
        else:
            height_axis = torch.tensor([0., 1., 0.])

        # Compute the 4 corners of the plate
        corners = [position, position + width_axis * self.width, position + height_axis * self.height,
                   position + width_axis * self.width + height_axis * self.height]

        # Return the concatenated corners
        return torch.cat(corners, dim=0)

    def resample(self, dt: float) -> 'RawMarkerPlateTrace':
        """
        This function returns a new RawMarkerPlateTrace with consistently spaced timesteps at the specified frequency.
        """
        # Create the new timesteps
        start_time = self.timesteps[0]
        end_time = self.timesteps[-1]
        new_timesteps = np.linspace(start_time, end_time, int((end_time - start_time) / dt) + 1)

        # Interpolators for positions, width_axis_direction, and height_axis_direction
        position_interpolator = interp1d(self.timesteps, np.array(self.positions), axis=0, kind='linear', fill_value='extrapolate')
        width_axis_interpolator = interp1d(self.timesteps, np.array(self.width_axis_direction), axis=0, kind='linear', fill_value='extrapolate')
        height_axis_interpolator = interp1d(self.timesteps, np.array(self.height_axis_direction), axis=0, kind='linear', fill_value='extrapolate')

        # Generate new interpolated values
        new_positions = position_interpolator(new_timesteps)
        new_width_axes = width_axis_interpolator(new_timesteps)
        new_height_axes = height_axis_interpolator(new_timesteps)

        # Renormalize the axes
        for i in range(len(new_positions)):
            new_width_axes[i] = new_width_axes[i] / np.linalg.norm(new_width_axes[i])
            new_height_axes[i] = new_height_axes[i] / np.linalg.norm(new_height_axes[i])
            new_height_axes[i] = new_height_axes[i] - np.dot(new_width_axes[i], new_height_axes[i]) * new_width_axes[i]
            new_height_axes[i] = new_height_axes[i] / np.linalg.norm(new_height_axes[i])

        # Create a new RawMarkerPlateTrace object
        new_trace = RawMarkerPlateTrace(self.width, self.height)
        new_trace.positions = new_positions.tolist()
        for i in range(len(new_trace.positions)):
            new_trace.positions[i] = np.array(new_trace.positions[i])
        new_trace.width_axis_direction = new_width_axes.tolist()
        for i in range(len(new_trace.width_axis_direction)):
            new_trace.width_axis_direction[i] = np.array(new_trace.width_axis_direction[i])
        new_trace.height_axis_direction = new_height_axes.tolist()
        for i in range(len(new_trace.height_axis_direction)):
            new_trace.height_axis_direction[i] = np.array(new_trace.height_axis_direction[i])
        new_trace.timesteps = new_timesteps.tolist()

        return new_trace

    def lowpass(self, cutoff_hz: float = 20.0) -> 'RawMarkerPlateTrace':
        """
        This function applies a low-pass filter to the plate position and orientation data.
        """
        # Create the new RawMarkerPlateTrace object
        new_trace = RawMarkerPlateTrace(self.width, self.height)
        new_trace.positions = self.positions.copy()
        new_trace.width_axis_direction = self.width_axis_direction.copy()
        new_trace.height_axis_direction = self.height_axis_direction.copy()
        new_trace.timesteps = self.timesteps.copy()

        # Apply the low-pass filter
        dt = np.mean(np.diff(new_trace.timesteps))
        nyquist = 0.5 / dt
        if cutoff_hz < nyquist:
            print('Applying low-pass filter with cutoff frequency', cutoff_hz, 'Hz')
            a, b = butter(1, cutoff_hz / nyquist, btype='low', analog=False, output='ba')
            new_trace.positions = filtfilt(a, b, new_trace.positions, axis=0)
            new_trace.width_axis_direction = filtfilt(a, b, new_trace.width_axis_direction, axis=0)
            new_trace.height_axis_direction = filtfilt(a, b, new_trace.height_axis_direction, axis=0)

            # Renormalize the axes
            for i in range(len(new_trace.positions)):
                if np.linalg.norm(new_trace.width_axis_direction[i]) > 0:
                    new_trace.width_axis_direction[i] = new_trace.width_axis_direction[i] / np.linalg.norm(new_trace.width_axis_direction[i])
                if np.linalg.norm(new_trace.height_axis_direction[i]) > 0:
                    new_trace.height_axis_direction[i] = new_trace.height_axis_direction[i] / np.linalg.norm(new_trace.height_axis_direction[i])
                    new_trace.height_axis_direction[i] = new_trace.height_axis_direction[i] - np.dot(new_trace.width_axis_direction[i], new_trace.height_axis_direction[i]) * new_trace.width_axis_direction[i]
                if np.linalg.norm(new_trace.height_axis_direction[i]) > 0:
                    new_trace.height_axis_direction[i] = new_trace.height_axis_direction[i] / np.linalg.norm(new_trace.height_axis_direction[i])
        else:
            print('Warning: Cutoff frequency is greater than Nyquist frequency. Skipping low-pass filter.')

        return new_trace