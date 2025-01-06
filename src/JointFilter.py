import numpy as np

from src.RelativeFilter import RelativeFilter
from .JointCenterProjector import JointCenterProjector
from .Board import Board, RawReading
class JointFilter:
    filter: RelativeFilter
    readings_projector: JointCenterProjector
    last_timestamp: float

    def __init__(self, parent_board: Board,
                 child_board: Board,
                 observability_threshold: float = 0.1,
                 acc_noise: np.ndarray = np.ones(3) * 0.05,
                 gyro_noise: np.ndarray = np.ones(3) * 0.05,
                 mag_noise: np.ndarray = np.ones(3) * 0.05):
        self.filter = RelativeFilter(acc_std=acc_noise, gyro_std=gyro_noise, mag_std=mag_noise)
        self.readings_projector = JointCenterProjector(parent_board=parent_board, child_board=child_board, observability_threshold=observability_threshold)
        self.last_timestamp = None

    def set_parent_and_child_state(self, q_wp, q_wc):
        self.filter.q_wp = q_wp
        self.filter.q_wc = q_wc

    def get_R_pc(self):
        return self.filter.get_R_pc()

    def update(self, raw_parent_reading: RawReading, raw_child_reading: RawReading):
        assert np.isclose(raw_parent_reading.timestamp, raw_child_reading.timestamp), "Parent and child readings must have the same timestamp."

        parent_reading, child_reading = self.readings_projector.get_projected_readings_at_joint_center(parent_reading=raw_parent_reading,
                                                                                                       child_reading=raw_child_reading)

        if self.last_timestamp is not None:
            dt = parent_reading.timestamp - self.last_timestamp
        else:
            dt = 0
        self.last_timestamp = parent_reading.timestamp

        self.filter.update(parent_reading.gyro, child_reading.gyro, parent_reading.acc, child_reading.acc, parent_reading.mag, child_reading.mag, dt)

    def to_dict(self):
        return {
            "gyro_noise": np.diagonal(self.filter.Q)[:3].tolist(),
            "acc_noise": np.diagonal(self.filter.R)[:3].tolist(),
            "mag_noise": np.diagonal(self.filter.R)[-3:].tolist(),
            "readings_projector": self.readings_projector.to_dict(),
        }

    @staticmethod
    def from_dict(joint_filter_dict):
        return JointFilter(parent_board=Board.from_dict(joint_filter_dict["readings_projector"]["parent_board"]),
                           child_board=Board.from_dict(joint_filter_dict["readings_projector"]["child_board"]),
                           observability_threshold=joint_filter_dict["readings_projector"]["observability_threshold"],
                           acc_noise=np.array(joint_filter_dict["acc_noise"]),
                           gyro_noise=np.array(joint_filter_dict["gyro_noise"]),
                           mag_noise=np.array(joint_filter_dict["mag_noise"]))
