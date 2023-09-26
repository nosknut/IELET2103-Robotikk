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
    z = np.linspace(-height/2, height/2, resolution)

    Phi, Z = np.meshgrid(phi, z)

    X = radius * np.cos(Phi)
    Y = radius * np.sin(Phi)

    return [X, Y, Z]


def plot_or_update_line(ax, xs, ys, zs, c=None, marker=None, line=None):
    if line:
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        return line
    else:
        l, = ax.plot(xs, ys, zs, c=c, marker=marker)
        return l


def plot_or_update_plane(ax, xs, ys, zs, c=None, line=None, alpha=None):
    if False and line:
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        return line
    else:
        l = ax.plot_surface(xs, ys, zs, color=c, alpha=alpha)
        return l


def draw_line(ax, start, end, c="r", marker="", line=None):
    xs, ys, zs = np.array([start, end]).T
    return plot_or_update_line(ax, xs, ys, zs, c=c, marker=marker, line=line)


def plot_axis(ax, lengths=[1, 1, 1]):
    x_length, y_length, z_length = lengths

    draw_line(ax, [-x_length, 0, 0], [x_length, 0, 0], c="r", marker="o"),
    draw_line(ax, [0, -y_length, 0], [0, y_length, 0], c="g", marker="o"),
    draw_line(ax, [0, 0, -z_length], [0, 0, z_length], c="b", marker="o"),


def rotate(matrix, rotation):
    r = R.from_euler('xyz', rotation)
    return matrix @ r.as_matrix()


def vec_length(vec):
    return np.sqrt(np.sum(np.square(vec)))


def draw_vector_cylinder(ax, vec, radius=0.1, line=None, r=[]):
    length = vec_length(vec)
    cylinder = meshgrid_to_matrix(create_cylinder(radius, length, 20))

    # Rotate cylinder to match vector
    x, y, z = r
    rotated = rotate(cylinder, [x, y, z])

    relocated = rotated  # + vec

    return plot_or_update_plane(ax, *matrix_to_meshgrid(relocated), c="r", alpha=0.5, line=line)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

rx = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'X', 0, np.pi, valinit=0)
ry = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'Y', 0, np.pi, valinit=0)
rz = Slider(plt.axes([0.25, 0.30, 0.65, 0.03]), 'Z', 0, np.pi, valinit=0)

la = None
lb = None

def draw():
    global la, lb

    vec = np.array([1, 1, 1])/2

    plot_axis(ax, [3, 3, 3])
    la = draw_line(ax, [rx.val, ry.val, rz.val], vec, c="orange", marker="o")
    lb = draw_vector_cylinder(ax, vec, r=[rx.val, ry.val, rz.val])

def update(val):
    ax.cla()
    draw()
    fig.canvas.draw_idle()

rx.on_changed(update)
ry.on_changed(update)
rz.on_changed(update)

draw()
plt.show()
