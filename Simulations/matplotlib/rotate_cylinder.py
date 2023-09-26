import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def meshgrid_to_matrix(meshgrid):
    X, Y, Z = meshgrid
    return np.matrix([X.flatten(), Y.flatten(), Z.flatten()]).T


def matrix_to_meshgrid(matrix):
    num_points = int(np.sqrt(matrix.shape[0]))
    X = matrix[:, 0].reshape((num_points, num_points))
    Y = matrix[:, 1].reshape((num_points, num_points))
    Z = matrix[:, 2].reshape((num_points, num_points))
    return [X, Y, Z]


def create_cylinder(radius, height, resolution=10):
    phi = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, resolution)

    Phi, Z = np.meshgrid(phi, z)

    X = radius * np.cos(Phi)
    Y = radius * np.sin(Phi)

    return [X, Y, Z]


def draw_line(ax, start, end, c="r", marker=""):
    xs, ys, zs = np.array([start, end]).T
    ax.plot(xs, ys, zs, c=c, marker=marker)


def plot_axis(ax, lengths=[1, 1, 1]):
    x_length, y_length, z_length = lengths

    draw_line(ax, [-x_length, 0, 0], [x_length, 0, 0], c="r", marker="o"),
    draw_line(ax, [0, -y_length, 0], [0, y_length, 0], c="g", marker="o"),
    draw_line(ax, [0, 0, -z_length], [0, 0, z_length], c="b", marker="o"),


def rotate(matrix, rotation):
    r = R.from_euler('zyx', rotation)
    return matrix @ r.as_matrix()


def vec_length(vec):
    return np.sqrt(np.sum(np.square(vec)))


def draw_vector_cylinder(ax, vec, radius=0.1, r=[]):
    length = vec_length(vec)
    cylinder = meshgrid_to_matrix(create_cylinder(radius, length, 10))

    # Rotate cylinder to match vector
    x, y, z = r
    rotated = rotate(cylinder, [z, y, x])

    relocated = rotated  + vec
    ax.plot_surface(*matrix_to_meshgrid(relocated), color="r", alpha=0.5)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

rx = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'X', 0, np.pi, valinit=0)
ry = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'Y', 0, np.pi, valinit=0)
rz = Slider(plt.axes([0.25, 0.30, 0.65, 0.03]), 'Z', 0, np.pi, valinit=0)


def draw():
    vec = np.array([1, 1, 1])

    plot_axis(ax, [2, 2, 2])
    # draw_line(ax, [0, 0, 0], vec, c="orange", marker="o")
    draw_vector_cylinder(ax, vec, r=[rx.val, ry.val, rz.val], radius=1)


def update(val):
    ax.cla()
    draw()
    fig.canvas.draw_idle()


rx.on_changed(update)
ry.on_changed(update)
rz.on_changed(update)

draw()
plt.show()
