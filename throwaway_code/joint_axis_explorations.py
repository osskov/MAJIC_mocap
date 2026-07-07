import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

# --- Helper Classes and Functions ---

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        super().draw(renderer)

def get_rotation_axis_from_svd(rotation_matrix):
    """
    Finds the eigenvector of a rotation matrix with eigenvalue 1.
    This is the axis of rotation.
    """
    # We are looking for a vector v such that Mv = v, or (M - I)v = 0.
    # This is the null space of the matrix (M - I).
    m_minus_i = rotation_matrix - np.identity(3)
    # SVD is a robust way to find the null space.
    # The right singular vector corresponding to the smallest singular value is our eigenvector.
    _, _, vh = np.linalg.svd(m_minus_i)
    # The eigenvector is the last row of Vh (or last column of V)
    axis = vh[-1]
    return axis

def draw_box(ax, vertices, color='cyan', alpha=0.5):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]], [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='r', alpha=alpha))

def draw_frame(ax, origin, rotation, scale=1.0, labels=['X', 'Y', 'Z']):
    colors = ['r', 'g', 'b']
    rotated_basis = rotation.apply(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale)
    for i in range(3):
        vec = rotated_basis[i]
        ax.add_artist(Arrow3D([origin[0], origin[0] + vec[0]], [origin[1], origin[1] + vec[1]], [origin[2], origin[2] + vec[2]], color=colors[i], mutation_scale=20, lw=1.5, arrowstyle="-|>"))

# --- Simulation Setup ---

# Define body K's geometry
vertices_k_initial = np.array([[-1,-2,-0.5], [1,-2,-0.5], [1,2,-0.5], [-1,2,-0.5], [-1,-2,0.5], [1,2,0.5], [1,2,0.5], [-1,2,0.5]])
# Define JOINT AXES in K's local frame
j1k_local = np.random.rand(3); j1k_local /= np.linalg.norm(j1k_local)
j2k_local = np.random.rand(3); j2k_local /= np.linalg.norm(j2k_local)

# Body K's random rotation parameters
k_rotation_axis = np.random.rand(3); k_rotation_axis /= np.linalg.norm(k_rotation_axis)
k_rotation_speed = 0.3
rotation_k = R.identity()

# Bodies J and L geometry and attachment points
vertices_small_box = np.array([[-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [0.5,0.5,-0.5], [-0.5,0.5,-0.5], [-0.5,-0.5,0.5], [0.5,-0.5,0.5], [0.5,0.5,0.5], [-0.5,0.5,0.5]])
translation_j_local = 2.5 * j1k_local
translation_l_local = 2.5 * j2k_local

# --- NEW: Constant Random Rotation for J and L (Sensor Noise/Drift) ---
rotation_j_noise = R.identity()
rotation_l_noise = R.identity()
noise_strength = 0.5 # Radians per animation step

# --- Buffers and Eigenvector Storage ---
BUFFER_SIZE = 15
rot_buffer_j = deque(maxlen=BUFFER_SIZE)
rot_buffer_l = deque(maxlen=BUFFER_SIZE)
eigenvector_j_local = None # Will store the calculated axis for J
eigenvector_l_local = None # Will store the calculated axis for L

    # --- ADDED: Apply continuous random rotation to J and L ---
j_noise_axis = np.random.rand(3) - 0.5; j_noise_axis /= np.linalg.norm(j_noise_axis)
l_noise_axis = np.random.rand(3) - 0.5; l_noise_axis /= np.linalg.norm(l_noise_axis)
rotation_j_noise = R.from_rotvec(noise_strength * j_noise_axis) * rotation_j_noise
rotation_l_noise = R.from_rotvec(noise_strength * l_noise_axis) * rotation_l_noise


# --- Matplotlib Setup ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

def update(frame_angle):
    global rotation_k, eigenvector_j_local, eigenvector_l_local, rotation_j_noise, rotation_l_noise
    ax.clear()
    ax.set_xlim([-5, 5]); ax.set_ylim([-5, 5]); ax.set_zlim([-5, 5])
    ax.set_xlabel('World X'); ax.set_ylabel('World Y'); ax.set_zlabel('World Z')
    ax.set_title("Eigenvector of Relative Rotation IS the Joint Axis")

    # Update and draw Body K
    delta_angle_k = k_rotation_speed * 0.05
    rotation_k = R.from_rotvec(delta_angle_k * k_rotation_axis) * rotation_k
    draw_box(ax, rotation_k.apply(vertices_k_initial), color='gray', alpha=0.1)
    draw_frame(ax, [0,0,0], rotation_k)

    # Transform local vectors to world frame
    translation_j_world = rotation_k.apply(translation_j_local)
    translation_l_world = rotation_k.apply(translation_l_local)
    j1k_world = rotation_k.apply(j1k_local)
    j2k_world = rotation_k.apply(j2k_local)

    # --- Body J Logic ---
    rotation_j_relative_to_k = R.from_rotvec(frame_angle * j1k_local)
    rotation_j_relative_to_k = rotation_j_relative_to_k * rotation_j_noise
    j1j_local = rotation_j_relative_to_k.inv().apply(j1k_local)
    # The total rotation of J now includes the noise component
    total_rotation_j = rotation_k * rotation_j_relative_to_k * rotation_j_noise
    draw_box(ax, total_rotation_j.apply(vertices_small_box) + translation_j_world, color='magenta', alpha=0.6)
    draw_frame(ax, translation_j_world, total_rotation_j)
    j1j_world = total_rotation_j.apply(j1j_local)

    # Draw the TRUE joint axis
    j1_end = translation_j_world + j1j_world * 2.0
    ax.add_artist(Arrow3D([translation_j_world[0], j1_end[0]], [translation_j_world[1], j1_end[1]], [translation_j_world[2], j1_end[2]], color='purple', mutation_scale=20, lw=3, arrowstyle="-|>"))
    ax.text(j1_end[0], j1_end[1], j1_end[2], 'True Axis', color='purple')

    # --- Body L Logic (Identical process) ---
    rotation_l_relative_to_k = R.from_rotvec(frame_angle * j2k_local)
    rotation_l_relative_to_k = rotation_l_relative_to_k * rotation_l_noise
    j2l_local = rotation_l_relative_to_k.inv().apply(j2k_local)
    # The total rotation of L now includes the noise component
    total_rotation_l = rotation_k * rotation_l_relative_to_k * rotation_l_noise
    draw_box(ax, total_rotation_l.apply(vertices_small_box) + translation_l_world, color='yellow', alpha=0.6)
    draw_frame(ax, translation_l_world, total_rotation_l)
    j2j_world = total_rotation_l.apply(j2l_local)

    # Draw the TRUE joint axis for L
    j2_end = translation_l_world + j2j_world * 2.0
    ax.add_artist(Arrow3D([translation_l_world[0], j2_end[0]], [translation_l_world[1], j2_end[1]], [translation_l_world[2], j2_end[2]], color='orange', mutation_scale=20, lw=3, arrowstyle="-|>"))

    j3k_local = np.cross(j1k_local, j2k_local)
    j3_fake_local = np.cross(j1j_local, j2l_local)

    ax.add_artist(Arrow3D([0, j3k_local[0]*3], [0, j3k_local[1]*3], [0, j3k_local[2]*3], color='green', mutation_scale=20, lw=3, arrowstyle="-|>"))
    ax.text(j3k_local[0]*3, j3k_local[1]*3, j3k_local[2]*3, 'Joint Axis 3 (K local)', color='green')
    ax.add_artist(Arrow3D([0, j3_fake_local[0]*3], [0, j3_fake_local[1]*3], [0, j3_fake_local[2]*3], color='blue', mutation_scale=20, lw=3, arrowstyle="-|>"))
    ax.text(j3_fake_local[0]*3, j3_fake_local[1]*3, j3_fake_local[2]*3, 'Joint Axis 3 (Fake local)', color='blue')

    j3_global = np.cross(total_rotation_j.apply(j1j_local), total_rotation_l.apply(j2l_local))
    ax.add_artist(Arrow3D([0, j3_global[0]*3], [0, j3_global[1]*3], [0, j3_global[2]*3], color='red', mutation_scale=20, lw=3, arrowstyle="-|>"))
    ax.text(j3_global[0]*3, j3_global[1]*3, j3_global[2]*3, 'Joint Axis 3 (Global)', color='red')

    # Plot the local versions of the vectors
    ax.text2D(0.05, 0.95, f"Joint Axis J (J local): {j1j_local}", transform=ax.transAxes)
    ax.text2D(0.05, 0.90, f"Joint Axis J (K local): {j1k_local}", transform=ax.transAxes)
    ax.text2D(0.05, 0.85, f"Joint Axis L (local): {j2l_local}", transform=ax.transAxes)
    ax.text2D(0.05, 0.80, f"Joint Axis L (K local): {j2k_local}", transform=ax.transAxes)
    ax.text2D(0.05, 0.75, f"Joint Axis 3 (Fake local): {j3_fake_local}", transform=ax.transAxes)
    ax.text2D(0.05, 0.70, f"Joint Axis 3 (K local): {j3k_local}", transform=ax.transAxes)
    ax.text2D(0.05, 0.65, f"Joint Axis 3 (Global): {j3_global}", transform=ax.transAxes)   

ani = FuncAnimation(fig, update, frames=np.linspace(0, 4 * np.pi, 300), interval=50)
plt.show()