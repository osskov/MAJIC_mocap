import os
import re
import time
from typing import List, Dict

import nimblephysics as nimble
import numpy as np
from scipy.interpolate import interp1d


def visualize_data(input_path: str):
    print('Loading C3D file: ', input_path)

    # Load the raw C3D
    c3d: nimble.biomechanics.C3D = nimble.biomechanics.C3DLoader.loadC3D(os.path.abspath(input_path))
    timestamps: List[float] = c3d.timestamps
    marker_timesteps: List[Dict[str, np.ndarray]] = c3d.markerTimesteps

    # timestamps = timestamps[:20000]
    # marker_timesteps = marker_timesteps[:20000]

    # Find the known marker names
    unknown_marker_regex = re.compile(r'U[0-9]+')
    marker_names = set()
    for marker_frame in marker_timesteps:
        marker_names.update(marker_frame.keys())
    unknown_markers = []
    known_markers = []
    for marker_name in marker_names:
        if unknown_marker_regex.match(marker_name):
            unknown_markers.append(marker_name)
        else:
            known_markers.append(marker_name)

    smoothed_trc_path = os.path.splitext(input_path)[0] + '_smoothed.trc'
    if os.path.isfile(smoothed_trc_path):
        smoothed_timesteps: List[Dict[str, np.ndarray]] = nimble.biomechanics.OpenSimParser.loadTRC(os.path.abspath(smoothed_trc_path)).markerTimesteps
    else:
        # Go through and interpolate and smooth and then re-gap the data
        dt = timestamps[1] - timestamps[0]
        smoothed_timesteps: List[Dict[str, np.ndarray]] = []
        for t, frame in enumerate(marker_timesteps):
            smoothed_frame = {}
            for marker in frame:
                smoothed_frame[marker] = np.array(frame[marker])
            smoothed_timesteps.append(smoothed_frame)

        smoother = nimble.utils.AccelerationMinimizer(len(marker_timesteps), 1.0 / (dt * dt), 10000.0)
        for marker in known_markers:
            print(f'AccelerationMinimizing marker: {marker}')
            velocities = [0.0]
            for t in range(1, len(marker_timesteps)):
                if marker in marker_timesteps[t] and marker in marker_timesteps[t - 1]:
                    velocity = np.linalg.norm(marker_timesteps[t][marker] - marker_timesteps[t - 1][marker]) / (
                            timestamps[t] - timestamps[t - 1])
                    velocities.append(velocity)
                else:
                    velocities.append(velocities[-1] if velocities else 0.0)
            accelerations = [0.0]
            for t in range(1, len(velocities)):
                acceleration = velocities[t] - velocities[t - 1]
                accelerations.append(acceleration)

            visible_timesteps = []
            visible_positions = []
            invisible_timesteps = []
            for t, frame in enumerate(marker_timesteps):
                if marker in frame and np.abs(accelerations[t]) < 0.2:
                    visible_timesteps.append(t)
                    visible_positions.append(frame[marker])
                else:
                    invisible_timesteps.append(t)
            # Interpolate the missing data
            for axis in range(3):
                filled_positions = np.zeros(len(marker_timesteps))
                interp = interp1d(visible_timesteps, [p[axis] for p in visible_positions], axis=0, kind='linear')
                for t in visible_timesteps:
                    filled_positions[t] = marker_timesteps[t][marker][axis]
                for t in invisible_timesteps:
                    if t >= min(visible_timesteps) and t <= max(visible_timesteps):
                        filled_positions[t] = interp(visible_timesteps[0])
                for t in range(max(visible_timesteps) + 1, len(marker_timesteps)):
                    filled_positions[t] = filled_positions[max(visible_timesteps)]
                for t in range(min(visible_timesteps) - 1, -1, -1):
                    filled_positions[t] = filled_positions[min(visible_timesteps)]
                # Smooth the data
                filled_positions = smoother.minimize(filled_positions)
                for t in range(min(visible_timesteps), max(visible_timesteps) + 1):
                    if marker in smoothed_timesteps[t]:
                        smoothed_timesteps[t][marker][axis] = filled_positions[t]

        # Save the smoothed data
        print('Saving smoothed data...')
        nimble.biomechanics.OpenSimParser.saveTRC(os.path.abspath(smoothed_trc_path), timestamps, smoothed_timesteps)

    print('Creating labelling GUI:')
    gui = nimble.NimbleGUI()
    gui.serve(8080)
    gui.nativeAPI().createText('frame', f'Frame: 0/{len(marker_timesteps)}', np.array([100, 200]), np.array([300, 50]))

    history_length = 50
    history_step_size = 1
    marker_size = np.array([0.04, 0.04, 0.04])
    tableau_20_colors_rgb = [
        (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
        (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
        (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
        (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
        (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
    ]
    tableau_20_colors = [np.array([r / 255, g / 255, b / 255, 1.0]) for r, g, b in tableau_20_colors_rgb]

    selected_marker: int = -1

    def select_marker(m: int):
        nonlocal selected_marker
        old_selected_marker_name = selected_marker
        gui.nativeAPI().setObjectScale(known_markers[old_selected_marker_name], marker_size)
        color = tableau_20_colors[m % len(tableau_20_colors)]
        gui.nativeAPI().setObjectColor(known_markers[old_selected_marker_name], color)

        selected_marker = m
        marker_name = known_markers[m]
        gui.nativeAPI().setObjectScale(marker_name, marker_size * 2.0)
        gui.nativeAPI().setObjectColor(marker_name, np.array([1.0, 0.0, 0.0, 1.0]))

    for m, marker in enumerate(known_markers):
        color = tableau_20_colors[m % len(tableau_20_colors)]
        gui.nativeAPI().createBox(marker, marker_size, np.zeros(3), np.zeros(3), color=color)
        gui.nativeAPI().setObjectTooltip(marker, marker)

    def render_frame(t: int):
        nonlocal selected_marker
        gui.nativeAPI().setTextContents('frame', f'Frame: {t}/{len(marker_timesteps)}')
        for m, marker in enumerate(known_markers):
            marker_pos_history = []
            for q in range(0, history_length, history_step_size):
                i = t - q
                if i < 0:
                    break
                if marker in marker_timesteps[i]:
                    marker_pos_history.append(smoothed_timesteps[i][marker])
            if len(marker_pos_history) == 0:
                marker_pos_history.append(np.array([0, 0, 0]))
            while len(marker_pos_history) < history_length:
                marker_pos_history.append(marker_pos_history[-1])
            color = tableau_20_colors[m % len(tableau_20_colors)] if m != selected_marker else np.array([1.0, 0.0, 0.0, 1.0])
            gui.nativeAPI().createLine(marker+'_line', marker_pos_history, color=color)
            gui.nativeAPI().setObjectPosition(marker, marker_pos_history[0])

    visualize_frame = 0
    playing = True

    def on_keydown(key: str):
        nonlocal visualize_frame
        nonlocal playing
        if key == ' ':
            playing = not playing
        if key == 'ArrowRight' or key == 'e' or key == 'd':
            visualize_frame += 1
            if visualize_frame >= len(timestamps):
                visualize_frame = 0
        if key == 'ArrowLeft' or key == 'a':
            visualize_frame -= 1
            if visualize_frame < 0:
                visualize_frame = len(timestamps) - 1
        render_frame(visualize_frame)
    gui.nativeAPI().registerKeydownListener(on_keydown)

    marker_rewrite_map: Dict[str, str] = {}
    for marker in known_markers:
        marker_rewrite_map[marker] = marker

    rewritten_frames = [marker_timesteps[0]]

    for scanning_frame in range(1, len(timestamps)):
        these_markers = marker_timesteps[scanning_frame]
        last_markers = marker_timesteps[scanning_frame - 1]

        for marker in known_markers:
            if marker in these_markers and marker in last_markers:
                jump_velocity = np.linalg.norm(these_markers[marker] - last_markers[marker]) / (timestamps[scanning_frame] - timestamps[scanning_frame - 1])
                if jump_velocity > 8.0:
                    print(f'Marker: {marker}, jump velocity: {jump_velocity} at frame {scanning_frame}')
                    select_marker(known_markers.index(marker))
                    visualize_frame = scanning_frame
                    render_frame(visualize_frame)
                    while True:
                        print('What should we rename this marker to?')
                        new_name = input()
                        if new_name in known_markers:
                            marker_rewrite_map[marker] = new_name
                            break
                        else:
                            print('Invalid marker name. Please choose from the following:')
                            print(known_markers)

        new_frame = {}
        for marker in these_markers:
            if marker in marker_rewrite_map:
                new_frame[marker_rewrite_map[marker]] = these_markers[marker]
        rewritten_frames.append(new_frame)

        if scanning_frame % 100 == 0:
            visualize_frame = scanning_frame
            render_frame(scanning_frame)

    gui.stopServing()

    gui.blockWhileServing()
    time.sleep(1.0)



if __name__ == '__main__':
    trial_path = '../../six_imu_data/raw_data/Subj1'
    # isolate the c3d file in this directory
    c3d_paths = [os.path.join(trial_path, f) for f in os.listdir(trial_path) if f.endswith('.c3d')]

    for c3d_path in c3d_paths:
        visualize_data(c3d_path)
