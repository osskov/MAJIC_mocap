from typing import Tuple, Dict
import numpy as np
from src.Board import Board, RawReading, ProjectedReading


class JointCenterProjector:
    parent_board: Board
    child_board: Board
    observability_threshold: float
    mag_choosing_method: str

    def __init__(self, parent_board: Board, child_board: Board, observability_threshold: float):
        self.parent_board = parent_board
        self.child_board = child_board
        self.observability_threshold = observability_threshold
        self.mag_choosing_method = None

    def __eq__(self, other: 'JointCenterProjector'):
        return (self.parent_board == other.parent_board and
                self.child_board == other.child_board and
                self.observability_threshold == other.observability_threshold)

    def get_projected_readings_at_joint_center(self, parent_reading: RawReading, child_reading: RawReading)\
            -> Tuple[ProjectedReading, ProjectedReading]:
        projected_parent_reading = self.parent_board.project_reading_to_joint_center(reading=parent_reading)
        projected_child_reading = self.child_board.project_reading_to_joint_center(reading=child_reading)
        averaged_parent_mag = np.mean(np.array(parent_reading.mags), axis=0)
        averaged_child_mag = np.mean(np.array(child_reading.mags), axis=0)

        if self.mag_choosing_method == "mag_free":
            projected_parent_reading.mag = np.zeros(3)
            projected_child_reading.mag = np.zeros(3)
            return projected_parent_reading, projected_child_reading
        elif self.mag_choosing_method == "average":
            projected_parent_reading.mag = averaged_parent_mag
            projected_child_reading.mag = averaged_child_mag
            return projected_parent_reading, projected_child_reading
        elif self.mag_choosing_method == "project":
            return projected_parent_reading, projected_child_reading
        else:
            parent_mag, child_mag = self._choose_mag_(projected_parent_reading, projected_child_reading,
                                                      averaged_parent_mag, averaged_child_mag)
            projected_parent_reading.mag = parent_mag
            projected_child_reading.mag = child_mag
            return projected_parent_reading, projected_child_reading

    def _choose_mag_(self, projected_parent_reading: ProjectedReading, projected_child_reading: ProjectedReading,
                     unprojected_parent_mag: np.ndarray, unprojected_child_mag: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        if self._check_observability_(projected_parent_reading, projected_child_reading) and self.mag_choosing_method is not 'best_mag':
            return np.zeros(3), np.zeros(3)
        else:
            unprojected_mag_norm_difference = np.abs(
                np.linalg.norm(unprojected_parent_mag) - np.linalg.norm(unprojected_child_mag))
            projected_mag_norm_difference = np.abs(
                np.linalg.norm(projected_parent_reading.mag) - np.linalg.norm(projected_child_reading.mag))
            if unprojected_mag_norm_difference < projected_mag_norm_difference:
                return unprojected_parent_mag, unprojected_child_mag
            else:
                return projected_parent_reading.mag, projected_child_reading.mag

    def _check_observability_(self, parent_reading: ProjectedReading, child_reading: ProjectedReading) -> bool:
        parent_observability = np.cross(parent_reading.acc_change, parent_reading.acc) / np.linalg.norm(
            parent_reading.acc)
        child_observability = np.cross(child_reading.acc_change, child_reading.acc) / np.linalg.norm(child_reading.acc)
        min_observability = min(np.linalg.norm(parent_observability), np.linalg.norm(child_observability))
        return min_observability > self.observability_threshold

    def to_dict(self) -> Dict[str, any]:
        return {
            "parent_board": self.parent_board.to_dict(),
            "child_board": self.child_board.to_dict(),
            "observability_threshold": self.observability_threshold
        }

    @staticmethod
    def from_dict(joint_center_projector_dict: Dict[str, any]) -> 'JointCenterProjector':
        parent_board = Board.from_dict(joint_center_projector_dict["parent_board"])
        child_board = Board.from_dict(joint_center_projector_dict["child_board"])
        return JointCenterProjector(parent_board=parent_board, child_board=child_board,
                                    observability_threshold=joint_center_projector_dict["observability_threshold"])
