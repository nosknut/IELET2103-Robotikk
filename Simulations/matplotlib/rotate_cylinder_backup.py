import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def create_circle(radius, resolution=10):
    thetas = np.linspace(0, 2*np.pi, resolution)
    xs = radius * np.cos(thetas)
    ys = radius * np.sin(thetas)

    # Uncomment to insert the points as readable text
    # This makes it easier to inspect what happens
    # to them during transformations later in the code

    # xs = ['x1', 'x2', 'x3']
    # ys = ['y1', 'y2', 'y3']

    # Shape: [[x1, x2], [y1, y2]]
    matrix = np.matrix([xs, ys])

    # Shape: [[x1, y1], [x2, y2]]
    return matrix.T

def add_axis(matrix):
    # matrix Shape: [[x1, x2], [y1, y2]]

    # Shape: [[x1, x2], [y1, y2]]
    transposed = list(matrix.T.A)

    # Shape: [[x1, x2], [y1, y2], [z1, z2]]
    transposed.append(np.ones(len(transposed[0])))

    # Shape: [[x1, y1, 1], [x2, y2, 1]]
    out_matrix = np.matrix(transposed).T

    return out_matrix

def create_cylinder(radius, height, resolution=10):
    xy_circle = create_circle(radius, resolution)

    xyz_circle = add_axis(xy_circle)

    # Shape: [[x1, y1, z1], [x2, y2, z2]]
    points = []

    zs = np.linspace(-height/2, height/2, 2)

    for z in zs:
        z_matrix = np.matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, z],
        ])

        p = xyz_circle @ z_matrix
        points = [*points, *list(p.A)]

    phi = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, resolution)
    Phi, Z = np.meshgrid(phi, z)
    print(Phi)
    X = radius * np.cos(Phi)
    Y = radius * np.sin(Phi)
    
    cylinder_points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    return np.matrix(cylinder_points)

    # Shape: [[x1, y1, z1], [x2, y2, z2]]
    return np.matrix(points)

def plot_axis(ax, lengths=[1, 1, 1]):
    line = np.linspace(-1, 1, 10)
    zeros = np.zeros(10)
    x_length, y_length, z_length = lengths

    ax.scatter(line * x_length, zeros, zeros, c='r', marker='o')
    ax.scatter(zeros, line * y_length, zeros, c='g', marker='o')
    ax.scatter(zeros, zeros, line * z_length, c='b', marker='o')

def rotate(matrix, rotation):
    r = R.from_rotvec(rotation)
    return matrix @ r.as_matrix()

# Matplotlib requires this format for ax.plot_surface()
def get_3d_meshgrid(matrix):
    xs, ys, zs = matrix.T.A
    cols = len(xs)
    rows = len(zs)

    # Shape: [[x1, x2, x3], [x1, x2, x3], [x1, x2, x3]]
    xm = np.array([xs for _ in range(rows)])
    
    # Shape: [[y1, y2, y3], [y1, y2, y3], [y1, y2, y3]]
    ym = np.array([ys for _ in range(rows)])
    
    # Shape: [[z1, z1, z1], [z2, z2, z2], [z3, z3, z3]]
    zm = np.array([np.array([z for _ in range(cols)]) for z in zs])
    
    return np.array([xm, ym, zm])

def to_mesh(matrix):
    print(matrix.shape[0])
    # Reshape points to meshgrid shape
    num_points = int(np.sqrt(matrix.shape[0]))
    print(num_points)
    X = matrix[:, 0].reshape((num_points, num_points))
    Y = matrix[:, 1].reshape((num_points, num_points))
    Z = matrix[:, 2].reshape((num_points, num_points))
    return [X, Y, Z]


def plot_matrix(ax, matrix, show_scatter = False):
    if show_scatter:
        xs, ys, zs = matrix.T.A
        ax.scatter(xs, ys, zs, c='r', marker='o')

    Xs, Ys, Zs = to_mesh(matrix)
    # ax.plot_surface(Xs, Ys, Zs)    
    ax.plot_surface(Xs, Ys, Zs, color='b', alpha=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cylinder = create_cylinder(1, 6, 50)

rotated = rotate(cylinder, np.array([0,0,0]))

# plot_matrix(ax, cylinder)
plot_matrix(ax, rotated)

plot_axis(ax, [2, 2, 2])

plt.show()
