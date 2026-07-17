import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace
from .PlateTrial import PlateTrial

# --- Parsers for raw files ---

def parse_imu_txt_file(file_path: str) -> IMUTrace:
    """Parses a single Xsens-formatted IMU .txt file."""
    freq = 100.0
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("// Update Rate"):
                try:
                    freq = float(line.split(":")[1].split("Hz")[0])
                except (IndexError, ValueError):
                    pass
                break
    df = pd.read_csv(file_path, delimiter='\t', skiprows=5)
    df = df.apply(pd.to_numeric)
    timestamps = 1 / freq * np.arange(len(df))
    acc = df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyro = df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    mag = df[['Mag_X', 'Mag_Y', 'Mag_Z']].values
    return IMUTrace(timestamps=timestamps, acc=acc, gyro=gyro, mag=mag)



def parse_trc_file(trc_file: str, max_trc_timestamp=-1.0, robust=True) -> Dict[str, WorldTrace]:
    """Parses a TRC file and extracts marker data into WorldTraces."""
    if trc_file is None or os.path.isfile(trc_file) is False or not trc_file.endswith('.trc'):
        raise FileNotFoundError("No TRC file found.")

    with open(trc_file, 'r') as file:
        lines = file.readlines()

    headers = lines[3].strip().split('\t')
    imu_headers = [header for header in headers if ('_O' in header or '_3' in header)]
    data = [line.strip().split('\t') for line in lines[6:]]  # Skip empty line and read the data
    data = np.array(data, dtype=float)
    timestamps = data[:, 1]

    if max_trc_timestamp > 0.0:
        first_exceeding_timestamp = np.argmax(timestamps > max_trc_timestamp)
        if first_exceeding_timestamp > 0:
            print(f"Trimming TRC data to {first_exceeding_timestamp} samples")
            data = data[:first_exceeding_timestamp]
            timestamps = timestamps[:first_exceeding_timestamp]

    assert len(np.unique(timestamps)) == len(timestamps), "Timestamps must be unique."
    world_traces = {}

    for i, imu_o_name in enumerate(imu_headers):
        if '_O' in imu_o_name:
            imu_o_idx = headers.index(imu_o_name)
            imu_x_idx = headers.index(imu_o_name.replace('_O', '_X'))
            imu_y_idx = headers.index(imu_o_name.replace('_O', '_Y'))
            imu_d_idx = headers.index(imu_o_name.replace('_O', '_D'))
        else:
            assert '_3' in imu_o_name
            imu_o_idx = headers.index(imu_o_name)
            imu_x_idx = headers.index(imu_o_name.replace('_3', '_2'))
            imu_y_idx = headers.index(imu_o_name.replace('_3', '_4'))
            imu_d_idx = headers.index(imu_o_name.replace('_3', '_1'))

        imu_o_loc = data[:, imu_o_idx: imu_o_idx + 3]
        imu_x_loc = data[:, imu_x_idx: imu_x_idx + 3]
        imu_y_loc = data[:, imu_y_idx: imu_y_idx + 3]
        imu_d_loc = data[:, imu_d_idx: imu_d_idx + 3]

        if "Foot" in imu_o_name:
            imu_o_copy = imu_o_loc.copy()
            imu_o_loc = imu_d_loc
            imu_d_loc = imu_o_copy
            if "R" in imu_o_name:
                imu_x_copy = imu_x_loc.copy()
                imu_x_loc = imu_d_loc
                imu_d_loc = imu_x_copy
                imu_y_copy = imu_y_loc.copy()
                imu_y_loc = imu_o_loc
                imu_o_loc = imu_y_copy
        if "L" in imu_o_name:
            imu_y_copy = imu_y_loc.copy()
            imu_y_loc = imu_d_loc
            imu_d_loc = imu_y_copy
            imu_x_copy = imu_x_loc.copy()
            imu_x_loc = imu_o_loc
            imu_o_loc = imu_x_copy

        if np.max(np.abs(imu_o_loc)) > 1000:
            imu_o_loc /= 1000
            imu_x_loc /= 1000
            imu_y_loc /= 1000
            imu_d_loc /= 1000

        if robust:
            world_traces[imu_o_name.replace('_O', '').replace('_3', '')] = WorldTrace.construct_from_markers_robust(
                timestamps, imu_o_loc, imu_d_loc, imu_x_loc, imu_y_loc
            )
        else:
            world_traces[imu_o_name.replace('_O', '').replace('_3', '')] = WorldTrace.construct_from_markers(
                timestamps, imu_o_loc, imu_d_loc, imu_x_loc, imu_y_loc
            )
    return world_traces

def parse_sto_file(sto_file: str) -> Dict[str, WorldTrace]:
    """Parses an OpenSim .sto file containing quaternion orientation data."""
    if not os.path.isfile(sto_file) or not sto_file.lower().endswith('.sto'):
        raise FileNotFoundError(f"No valid .sto file found at path: {sto_file}")

    with open(sto_file, 'r') as f:
        lines = f.readlines()

    try:
        header_end_index = next(i for i, line in enumerate(lines) if 'endheader' in line)
    except StopIteration:
        raise ValueError("'.sto' file is missing the 'endheader' line.")

    column_headers = lines[header_end_index + 1].strip().split('\t')
    try:
        time_col_idx = column_headers.index('time')
    except ValueError:
        raise ValueError("'.sto' file is missing the 'time' column.")
    
    imu_headers = [h for h in column_headers if h != 'time']
    if not imu_headers:
        raise ValueError("No data columns found in the .sto file besides 'time'.")

    data_lines = lines[header_end_index + 2:]
    timestamps = []
    imu_quat_data = {header: [] for header in imu_headers}

    for line in data_lines:
        if not line.strip(): continue
        parts = line.strip().split('\t')
        timestamps.append(float(parts[time_col_idx]))
        for header in imu_headers:
            header_idx = column_headers.index(header)
            quat_str = parts[header_idx]
            w, x, y, z = [float(v) for v in quat_str.split(',')]
            imu_quat_data[header].append([x, y, z, w])

    world_traces = {}
    timestamps_np = np.array(timestamps)
    num_frames = len(timestamps_np)
    positions = np.zeros((num_frames, 3))

    for header, quats in imu_quat_data.items():
        rotation_matrices = Rotation.from_quat(quats).as_matrix()
        world_traces[header] = WorldTrace(timestamps=timestamps_np, positions=positions, rotations=rotation_matrices)
    
    return world_traces

def _load_imu_traces_from_structure(imu_folder_path: str, data_subdirectory_parts: List[str]) -> Dict[str, IMUTrace]:
    """Generic helper to load IMU traces based on XML mapping."""
    imu_traces = {}
    mapping_file = next((f for f in os.listdir(imu_folder_path) if f.endswith('.xml')), None)
    if mapping_file is None:
        raise FileNotFoundError(f"No mapping file (.xml) found in IMU folder: {imu_folder_path}")

    tree = ET.parse(os.path.join(imu_folder_path, mapping_file))
    root = tree.getroot()
    trial_prefix_element = root.find('.//trial_prefix')
    trial_prefix = trial_prefix_element.text if trial_prefix_element is not None else ""

    for sensor in root.findall('.//ExperimentalSensor'):
        sensor_name = sensor.get('name').strip()
        name_in_model = sensor.find('name_in_model').text.strip()
        file_name = f"{trial_prefix}{sensor_name}.txt"
        file_path = os.path.join(imu_folder_path, *data_subdirectory_parts, file_name)
        try:
            imu_traces[name_in_model] = parse_imu_txt_file(file_path)
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found.")
    
    return imu_traces

def _load_world_traces_from_mapping_file_and_folder(folder_path: str, mapping_file: str) -> Dict[str, WorldTrace]:
    """Generic helper to load Madgwick WorldTraces based on XML mapping."""
    world_traces = {}
    tree = ET.parse(mapping_file)
    root = tree.getroot()
    trial_prefix_elem = root.find('.//trial_prefix')
    if trial_prefix_elem is None or trial_prefix_elem.text is None:
        raise ValueError("Mapping XML missing 'trial_prefix' element or its text.")
    trial_prefix = trial_prefix_elem.text.strip()

    for sensor in root.findall('.//ExperimentalSensor'):
        name_attr = sensor.get('name')
        if name_attr is None: continue
        sensor_name = name_attr.strip()
        name_in_model_elem = sensor.find('name_in_model')
        if name_in_model_elem is None or name_in_model_elem.text is None: continue
        name_in_model = name_in_model_elem.text.strip()
        
        file_name = f"{trial_prefix}{sensor_name}.txt"
        file_path = os.path.join(folder_path, file_name)
        freq = 100.0
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if line.startswith("// Update Rate"):
                        freq = float(line.split(":")[1].split("Hz")[0])
                        if "Subject06" in folder_path or "Subject10" in folder_path:
                            freq = 40.0
                        break
            df = pd.read_csv(file_path, delimiter='\t', skiprows=5)
            df = df.apply(pd.to_numeric)
            timestamps = 1 / freq * np.arange(len(df))
            if df['Mat[3][3]'].isna().any():
                rotations = df[['Mag_Z','Mat[3][1]', 'Mat[3][2]', 'Mat[1][1]', 'Mat[1][2]', 'Mat[1][3]', 'Mat[2][1]', 'Mat[2][2]', 'Mat[2][3]']].values.reshape(-1, 3, 3)
            else:
                rotations = df[['Mat[1][1]', 'Mat[1][2]', 'Mat[1][3]', 'Mat[2][1]', 'Mat[2][2]', 'Mat[2][3]', 'Mat[3][1]', 'Mat[3][2]', 'Mat[3][3]']].values.reshape(-1, 3, 3)
            
            world_traces[name_in_model] = WorldTrace(timestamps=timestamps, positions=np.zeros((len(timestamps), 3)), rotations=rotations)
            if freq < 100.0:
                world_traces[name_in_model] = world_traces[name_in_model].resample(100.0)
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping sensor {sensor_name}.")
    
    return world_traces

# --- Loaders ---

class BaseDatasetLoader:

    def __init__(self, folder_path: str, robust: bool = True):
        self.folder_path = folder_path
        self.robust = robust

    def load_imu_traces(self) -> Dict[str, IMUTrace]:
        raise NotImplementedError

    def load_world_traces(self) -> Dict[str, WorldTrace]:
        raise NotImplementedError

    def load_plate_trials(self, align_plate_trials: bool = True) -> List[PlateTrial]:
        # Standard name map for many of the datasets
        IMU_TO_TRC_NAME_MAP = {
            'pelvis_imu': 'Pelvis_IMU', 'femur_r_imu': 'R.Femur_IMU', 'femur_l_imu': 'L.Femur_IMU',
            'tibia_r_imu': 'R.Tibia_IMU', 'tibia_l_imu': 'L.Tibia_IMU', 'calcn_r_imu': 'R.Foot_IMU',
            'calcn_l_imu': 'L.Foot_IMU', 'torso_imu': 'Back_IMU'
        }

        imu_traces = self.load_imu_traces()
        world_traces = self.load_world_traces()
        
        # Rename world traces to match IMU traces based on the name map
        renamed_world_traces = {}
        # Create a reverse map to go from TRC name to IMU name
        trc_to_imu_map = {v: k for k, v in IMU_TO_TRC_NAME_MAP.items()}
        
        for k, v in world_traces.items():
            if k in trc_to_imu_map:
                renamed_world_traces[trc_to_imu_map[k]] = v
            else:
                renamed_world_traces[k] = v
                
        return PlateTrial.generate_plate_from_traces(
            imu_traces, renamed_world_traces, align_plate_trials
        )

class AlBornoLoader(BaseDatasetLoader):
    def load_imu_traces(self) -> Dict[str, IMUTrace]:
        return _load_imu_traces_from_structure(self.folder_path, ['xsens', 'LowerExtremity'])

    def load_world_traces(self) -> Dict[str, WorldTrace]:
        # Tries to load from TRC or Madgwick depending on what was historically done.
        mocap_folder = os.path.join(self.folder_path, 'Mocap/')
        if os.path.isdir(mocap_folder):
            trc_files = [file for file in os.listdir(mocap_folder) if file.endswith('.trc') and 'static' not in file]
            if trc_files:
                trc_file_path = os.path.abspath(os.path.join(mocap_folder, trc_files[0]))
                return parse_trc_file(trc_file_path, robust=self.robust)

        # Fallback to Madgwick loader if no TRC
        mapping_file = next((f for f in os.listdir(self.folder_path) if f.endswith('.xml')), None)
        mapping_file_path = os.path.join(self.folder_path, mapping_file) if mapping_file else None
        madgwick_path = os.path.join(self.folder_path, 'madgwick', 'LowerExtremity')
        if mapping_file_path and os.path.isdir(madgwick_path):
            return _load_world_traces_from_mapping_file_and_folder(madgwick_path, mapping_file_path)
            
        raise FileNotFoundError("Could not find valid TRC or Madgwick data in Al Borno folder.")

class SkovLoader(BaseDatasetLoader):
    def load_imu_traces(self) -> Dict[str, IMUTrace]:
        return _load_imu_traces_from_structure(self.folder_path, ['imu data'])

    def load_world_traces(self) -> Dict[str, WorldTrace]:
        trc_files = [file for file in os.listdir(self.folder_path) if file.endswith('.trc')]
        if not trc_files:
            raise FileNotFoundError(f"No .trc file found in {self.folder_path}")
        trc_file_path = os.path.abspath(os.path.join(self.folder_path, trc_files[0]))
        return parse_trc_file(trc_file_path, robust=self.robust)

# Backwards compatibility helper
def load_trial_from_folder(folder_path: str, align_plate_trials=True, robust=True) -> List[PlateTrial]:
    # Use SkovLoader as the default for the old load_trial_from_folder
    return SkovLoader(folder_path, robust=robust).load_plate_trials(align_plate_trials=align_plate_trials)
