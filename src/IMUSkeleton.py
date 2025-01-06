import json
from typing import List, Dict, Tuple
from src.toolchest.PlateTrial import PlateTrial
import numpy as np
from .JointFilter import JointFilter, RawReading, Board
class Joint:
    joint_filter: JointFilter
    parent_name: str
    R_pb: np.ndarray
    child_name: str
    R_cb: np.ndarray

    def __init__(self, parent_name: str,
                 child_name: str,
                 joint_filter: JointFilter,
                 R_pb: np.ndarray = np.eye(3),
                 R_cb: np.ndarray = np.eye(3)):
        self.parent_name = parent_name
        self.child_name = child_name
        self.joint_filter = joint_filter
        self.R_pb = R_pb
        self.R_cb = R_cb

    def to_dict(self) -> Dict[str, any]:
        return {
            "parent_name": self.parent_name,
            "child_name": self.child_name,
            "joint_filter": self.joint_filter.to_dict(),
            "R_pb": self.R_pb.tolist(),
            "R_cb": self.R_cb.tolist()
        }

    @staticmethod
    def from_dict(joint_dict: Dict) -> 'Joint':
        return Joint(parent_name=joint_dict["parent_name"],
                     child_name=joint_dict["child_name"],
                     joint_filter=JointFilter.from_dict(joint_dict["joint_filter"]),
                     R_pb=np.array(joint_dict["R_pb"]),
                     R_cb=np.array(joint_dict["R_cb"])
                     )

class IMUSkeleton:
    joints: Dict[str, Joint]

    def __init__(self, joints: Dict[str, Joint]):
        self.joints = joints

    @staticmethod
    def load_from_json(json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        joints = {joint_name: Joint.from_dict(joint_data) for joint_name, joint_data in data.items()}
        return IMUSkeleton(joints)

    @staticmethod
    def create_json_from_pilot_plate_trials(folder: str):
        # load plate trials
        plate_trials = PlateTrial.load_cheeseburger_trial_from_folder(folder)
        # generate skeleton
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

        skeleton_dict = {}

        for joint_name, segment_tuple in joint_segment_dict.items():
            # Build the parent and child boards
            # get the parent offset from joint center then add the local offsets
            parent_name = segment_tuple[0]
            child_name = segment_tuple[-1]

            parent_trial = next(plate_trial for plate_trial in plate_trials if plate_trial.name.__contains__(parent_name))
            child_trial = next(plate_trial for plate_trial in plate_trials if plate_trial.name.__contains__(child_name))

            parent_offset, child_offset, _ = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
            parent_offsets = [parent_offset + parent_trial.imu_offset, parent_offset + parent_trial.second_imu_offset]
            child_offsets = [child_offset + child_trial.imu_offset, child_offset + child_trial.second_imu_offset]

            parent_board = Board(parent_offsets, parent_offsets)
            child_board = Board(child_offsets, child_offsets)
            observability_threshold = 0.1
            # Generate joint filter
            joint_filter = JointFilter(parent_board=parent_board, child_board=child_board, observability_threshold=observability_threshold)
            # Generate joint
            skeleton_dict[joint_name] = Joint(parent_name, child_name, joint_filter)

        skeleton = IMUSkeleton(skeleton_dict)
        skeleton.save_to_json(folder + '/skeleton.json')

    def save_to_json(self, json_path: str):
        data = {joint_name: joint.to_dict() for joint_name, joint in self.joints.items()}

        # Ensure the folder exists
        folder_path = '/'.join(json_path.split('/')[:-1])
        import os
        os.makedirs(folder_path, exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(data, f)

    def update(self, raw_readings: Dict[str, RawReading]):
        for joint_name, joint in self.joints.items():
            parent_reading = next(raw_readings[parent_name] for parent_name in raw_readings if parent_name.__contains__(joint.parent_name))
            child_reading = next(raw_readings[child_name] for child_name in raw_readings if child_name.__contains__(joint.child_name))
            joint.joint_filter.update(parent_reading, child_reading)

    def get_joint_angles(self) -> Dict[str, np.ndarray]:
        return {joint_name: joint.R_pb.T @ joint.joint_filter.get_R_pc() @ joint.R_cb for joint_name, joint in self.joints.items()}

    def get_joint_dict(self) -> Dict[str, Tuple[str, str]]:
        return {joint_name: (joint.parent_name, joint.child_name) for joint_name, joint in self.joints.items()}