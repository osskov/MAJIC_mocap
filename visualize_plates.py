import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from typing import List, Optional, Tuple, Dict
import os

# --- Import your actual classes here ---
# Ensure that PlateTrial.py, IMUTrace.py, and WorldTrace.py are in your Python path
# or in the same directory as this script.
# Assuming IMUTrace and WorldTrace are imported within PlateTrial.py or are in the same directory.
from src.toolchest.PlateTrial import PlateTrial


def visualize_plates_with_slider(
        plate_trials: List[PlateTrial],
        axis_length: float = 0.05,  # Length of the local axes for visualization
        colors: Optional[List[str]] = None
):
    """
    Visualizes a list of PlateTrial objects in 3D space with a time slider and play button,
    showing their world_trace positions (trajectories) and rotations (local coordinate systems).

    Args:
        plate_trials (List[PlateTrial]): A list of PlateTrial objects to visualize.
        axis_length (float): The length of the X, Y, Z axes drawn for rotations.
        colors (Optional[List[str]]): A list of colors for each plate trial. If None,
                                      default matplotlib colors will be used.
    """
    if not plate_trials:
        print("No PlateTrial objects provided for visualization.")
        return

    # Get the total number of timesteps from the first plate trial
    # Assuming all plate trials have the same length
    num_timesteps = len(plate_trials[0].world_trace.positions)
    if num_timesteps == 0:
        print("PlateTrial has no data points to visualize.")
        return

    # Setup the matplotlib figure and 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Adjust plot to make space for the slider and button at the bottom
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Set up colors for trajectories and triads
    if colors is None:
        default_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
        colors = [default_colors[i % len(default_colors)] for i in range(len(plate_trials))]
    elif len(colors) < len(plate_trials):
        print("Warning: Not enough colors provided for all plate trials. Recycling colors.")
        extended_colors = list(colors)
        default_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
        for i in range(len(plate_trials) - len(colors)):
            extended_colors.append(default_colors[(len(colors) + i) % len(default_colors)])
        colors = extended_colors

    # Store quiver objects for updating
    # Each element in quiver_objects will be a dictionary:
    # {'x': QuiverObject, 'y': QuiverObject, 'z': QuiverObject, 'trial_idx': int}
    quiver_objects = []

    # Plot full trajectories and initialize quiver objects for the first timestep
    for idx, trial in enumerate(plate_trials):
        # Basic check for required attributes
        if not hasattr(trial, 'world_trace') or \
                not hasattr(trial.world_trace, 'positions') or \
                not hasattr(trial.world_trace, 'rotations') or \
                len(trial.world_trace.positions) == 0:
            print(f"Skipping PlateTrial '{trial.name}' due to missing or empty 'world_trace' data.")
            continue

        # Get all positions for the trajectory plot
        all_positions = np.array(trial.world_trace.positions)
        # Plot trajectory with 0.05 opacity
        ax.plot(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2],
                label=f'{trial.name} Trajectory', color=colors[idx], linestyle='--', alpha=0.05)

        # Initialize quiver objects for the first timestep (timestep 0)
        initial_pos = all_positions[0, :]
        initial_rot_matrix = np.array(trial.world_trace.rotations[0])

        # Calculate initial axis vectors in world frame
        x_axis_world_initial = initial_rot_matrix @ np.array([1, 0, 0]) * axis_length
        y_axis_world_initial = initial_rot_matrix @ np.array([0, 1, 0]) * axis_length
        z_axis_world_initial = initial_rot_matrix @ np.array([0, 0, 1]) * axis_length

        # Create initial quiver objects. These will be updated by the slider.
        q_x = ax.quiver(initial_pos[0], initial_pos[1], initial_pos[2],
                        x_axis_world_initial[0], x_axis_world_initial[1], x_axis_world_initial[2],
                        color='r', length=axis_length, arrow_length_ratio=0.3, linewidth=1.5)
        q_y = ax.quiver(initial_pos[0], initial_pos[1], initial_pos[2],
                        y_axis_world_initial[0], y_axis_world_initial[1], y_axis_world_initial[2],
                        color='g', length=axis_length, arrow_length_ratio=0.3, linewidth=1.5)
        q_z = ax.quiver(initial_pos[0], initial_pos[1], initial_pos[2],
                        z_axis_world_initial[0], z_axis_world_initial[1], z_axis_world_initial[2],
                        color='b', length=axis_length, arrow_length_ratio=0.3, linewidth=1.5)
        quiver_objects.append({'x': q_x, 'y': q_y, 'z': q_z, 'trial_idx': idx})

    # Set plot labels, title, legend, and grid
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Visualization of PlateTrial Trajectories and Orientations')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')  # 'auto' for motion data, 'equal' if true aspect ratio is critical

    # Create the slider axis
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    # Create the slider, ranging from 0 to num_timesteps-1, starting at 0, with integer steps
    slider = Slider(ax_slider, 'Time Step', 0, num_timesteps - 1, valinit=0, valstep=1)

    # Global variables for animation control
    is_playing = False
    ani = None # Will hold the FuncAnimation object

    def update_quiver_objects(timestep: int):
        """
        Updates the positions and orientations of the triads for a given timestep.
        """
        # Update each set of quiver objects (triads) for each plate trial
        for q_obj in quiver_objects:
            trial_idx = q_obj['trial_idx']
            trial = plate_trials[trial_idx]

            # Get the position and rotation matrix for the current timestep
            current_pos = np.array(trial.world_trace.positions[timestep])
            current_rot_matrix = np.array(trial.world_trace.rotations[timestep])

            # Calculate the current axis vectors in the world frame
            x_axis_world = current_rot_matrix @ np.array([1, 0, 0]) * axis_length
            y_axis_world = current_rot_matrix @ np.array([0, 1, 0]) * axis_length
            z_axis_world = current_rot_matrix @ np.array([0, 0, 1]) * axis_length

            # Update the segments of the quiver objects
            q_obj['x'].set_segments([[[current_pos[0], current_pos[1], current_pos[2]],
                                      [current_pos[0] + x_axis_world[0], current_pos[1] + x_axis_world[1],
                                       current_pos[2] + x_axis_world[2]]]])
            q_obj['y'].set_segments([[[current_pos[0], current_pos[1], current_pos[2]],
                                      [current_pos[0] + y_axis_world[0], current_pos[1] + y_axis_world[1],
                                       current_pos[2] + y_axis_world[2]]]])
            q_obj['z'].set_segments([[[current_pos[0], current_pos[1], current_pos[2]],
                                      [current_pos[0] + z_axis_world[0], current_pos[1] + z_axis_world[1],
                                       current_pos[2] + z_axis_world[2]]]])

        # Redraw the canvas to reflect the changes
        fig.canvas.draw_idle()

    def update_slider_and_quiver(frame):
        """
        Function called by FuncAnimation to update the slider and quiver objects.
        """
        slider.set_val(frame) # This will trigger the slider.on_changed(update_quiver_objects) callback
        # FuncAnimation expects an iterable of artists that were modified.
        # Since update_quiver_objects modifies them in place, we can return them.
        all_quivers = [q_obj['x'] for q_obj in quiver_objects] + \
                      [q_obj['y'] for q_obj in quiver_objects] + \
                      [q_obj['z'] for q_obj in quiver_objects]
        return all_quivers

    def toggle_animation(event):
        """
        Callback function for the Play/Pause button.
        Toggles the animation on and off.
        """
        nonlocal is_playing, ani
        if not is_playing:
            # Start animation
            is_playing = True
            button_play.label.set_text('Pause')
            # Create FuncAnimation. interval is in milliseconds. repeat=False means it stops at the end.
            ani = animation.FuncAnimation(fig, update_slider_and_quiver,
                                          frames=num_timesteps,
                                          interval=50, # 50ms interval = 20 frames per second
                                          repeat=False,
                                          blit=False) # blitting can be tricky with 3D plots
            fig.canvas.draw_idle()
        else:
            # Pause animation
            is_playing = False
            button_play.label.set_text('Play')
            if ani:
                ani.event_source.stop()
            fig.canvas.draw_idle()

    # Register the update function to be called when the slider's value changes
    slider.on_changed(update_quiver_objects)

    # Create the Play/Pause button
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.04]) # [left, bottom, width, height]
    button_play = Button(ax_button, 'Play', color='lightgreen', hovercolor='0.975')
    button_play.on_clicked(toggle_animation)

    plt.show()


if __name__ == "__main__":
    # --- Example Usage ---
    # This section demonstrates how you would load your actual PlateTrial objects.
    # You will need to replace the placeholder comments with your actual data loading logic.

    print("Loading plate trials for visualization...")

    # Example: Load a list of PlateTrial objects from your data source.
    # This might involve calling a function from your PlateTrial class
    # or iterating through a directory of data files.
    #
    # For instance, if your PlateTrial class has a method like:
    # PlateTrial.load_from_data_source(path_to_data) -> List[PlateTrial]
    #
    # You would use it like this:
    # all_plate_trials = PlateTrial.load_from_data_source("path/to/your/data")

    # The user provided this line for loading data:


    all_plate_trials: List[PlateTrial] = PlateTrial.load_trial_from_folder("data/ODay_Data/Subject03/walking", False)
    all_plate_trials = [trial[:100] for trial in all_plate_trials]  # Limit to first 100 timesteps for faster visualization
    # for i in range(11):
    #     subject = f"Subject{i+1:02d}"
    #     # The user provided this line for loading data:
    #     all_plate_trials: List[PlateTrial] = PlateTrial.load_trial_from_folder(f"test_data/oday_data/{subject}/walking")
    print("done")
    if all_plate_trials:
        visualize_plates_with_slider(all_plate_trials)
    # else:
    #     print("No plate trials loaded. Please ensure your data loading logic is correct.")

