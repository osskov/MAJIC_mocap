import json
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

import nimblephysics as nimble
import numpy as np

input_path = '../../test_data/Pilot_IMU_Data/osim_results.b3d'
mapping_path = '../../test_data/Pilot_IMU_Data/mapping.json'
output_path = '../../test_data/Pilot_IMU_Data/'

# Load the mapping
with open(mapping_path, 'r') as f:
    mapping = json.load(f)

# Get the names of the IMU plates
imu_names = []
for values in mapping.values():
    if values[-2:] == '_1' or values[-2:] == '_2':
        if values[:-2] not in imu_names:
            imu_names.append(values[:-2])

# Load the raw subject on disk
subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(os.path.abspath(input_path))

raw_osim = subject.getOpensimFileText(subject.getNumProcessingPasses() - 1)

osim_file = subject.readOpenSimFile(subject.getNumProcessingPasses() - 1, ignoreGeometry=True)

marker_map: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]] = osim_file.markersMap


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # For some reason, numpy's cross product causes the IDE to freak out and say that any code after it is
    # unreachable.
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


# Compute the frames for each IMU on the subject
body_frames: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
for imu_name in imu_names:
    body, corner_offset = marker_map[imu_name + '_2']
    x_axis = marker_map[imu_name + '_1'][1] - corner_offset
    x_axis /= np.linalg.norm(x_axis)
    y_axis = corner_offset - marker_map[imu_name + '_3'][1]
    y_axis /= np.linalg.norm(y_axis)
    z_axis = cross(x_axis, y_axis)
    y_axis = cross(z_axis, x_axis)
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    euler_angles = nimble.math.matrixToEulerXYZ(rotation_matrix)

    imu_center = (marker_map[imu_name + '_2'][1] + marker_map[imu_name + '_4'][1]) / 2

    if body.getName() not in body_frames:
        body_frames[body.getName()] = []
    body_frames[body.getName()].append((imu_name, imu_center, euler_angles))

# Parse the XML string
root = ET.fromstring(raw_osim)

# Assert that the root node is <OpenSimDocument>
assert root.tag == 'OpenSimDocument', f"Root tag is {root.tag}, expected 'OpenSimDocument'"

model = root.find('Model')
assert model is not None, "No <Model> element found"

# Find the <BodySet> child
body_set = model.find('BodySet')
assert body_set is not None, "No <BodySet> element found"

if body_set.find('objects') is not None:
    body_set = body_set.find('objects')

# Iterate over each <Body> element
for body in body_set.findall('Body'):
    name = body.get('name')
    if name in body_frames:
        components = ET.SubElement(body, 'components')
        for imu_name, imu_center, euler_angles in body_frames[name]:
            print(f'Adding frame for {imu_name} on {name}')

            # Add the IMU frame
            child_frame = ET.SubElement(components, 'PhysicalOffsetFrame')
            child_frame.set('name', imu_name)

            # Add the frame geometry (axis for visualizer)
            frame_geometry = ET.SubElement(child_frame, 'FrameGeometry')
            frame_geometry.set('name', 'frame_geometry')

            frame_geometry_socket = ET.SubElement(frame_geometry, 'socket_frame')
            frame_geometry_socket.text = '..'

            frame_geometry_scale = ET.SubElement(frame_geometry, 'scale_factors')
            frame_geometry_scale.text = '0.2 0.2 0.2'

            # Add the bricks for visualizer
            attached_geometry = ET.SubElement(child_frame, 'attached_geometry')
            brick = ET.SubElement(attached_geometry, 'Brick')
            brick.set('name', imu_name + '_geom')

            brick_socket_frame = ET.SubElement(brick, 'socket_frame')
            brick_socket_frame.text = '..'

            brick_appearance = ET.SubElement(brick, 'Appearance')
            brick_color = ET.SubElement(brick_appearance, 'color')
            brick_color.text = '0.5 0.5 1'

            brick_half_lengths = ET.SubElement(brick, 'half_lengths')
            brick_half_lengths.text = '0.02 0.01 0.005'

            # Add the frame properties
            socket_parent = ET.SubElement(child_frame, 'socket_parent')
            socket_parent.text = '..'

            translation = ET.SubElement(child_frame, 'translation')
            translation.text = f'{imu_center[0]} {imu_center[1]} {imu_center[2]}'

            orientation = ET.SubElement(child_frame, 'orientation')
            orientation.text = f'{euler_angles[0]} {euler_angles[1]} {euler_angles[2]}'

# Write the updated XML tree to a string
ET.indent(root, space="  ", level=0)
updated_xml_string = ET.tostring(root, encoding='unicode')
with open(os.path.join(output_path, 'scaled_with_imus.osim'), 'w') as f:
    f.write(updated_xml_string)

# Create a configuration XML
configuration = ET.Element('OpenSimDocument')
reader_settings = ET.SubElement(configuration, 'XSensDataReaderSettings')
trial_prefix = ET.SubElement(reader_settings, 'trial_prefix')
trial_prefix.text = 'random_motion_'
experimental_sensors = ET.SubElement(reader_settings, 'ExperimentalSensors')
for key in mapping:
    if mapping[key][-2:] != '_1':
        continue
    sensor = ET.SubElement(experimental_sensors, 'ExperimentalSensor')
    sensor.set('name', '_' + key)
    name_in_model = ET.SubElement(sensor, 'name_in_model')
    name_in_model.text = mapping[key][:-2]

# Write the configuration XML to a string
# Indent the XML tree
ET.indent(configuration, space="  ", level=0)

configuration_string = ET.tostring(configuration, encoding='unicode')
with open(os.path.join(output_path, 'opensense_configuration.xml'), 'w') as f:
    f.write(configuration_string)
