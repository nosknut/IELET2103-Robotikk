import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def create_circle(radius):
    thetas = np.linspace(0, 2*np.pi, 100)
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


def create_cylinder(radius, height):
    xy_circle = create_circle(radius)

    xyz_circle = add_axis(xy_circle)

    # Shape: [[x1, y1, z1], [x2, y2, z2]]
    points = []

    zs = np.linspace(-height/2, height/2, 2)

    for z in zs:
        z_matrix = np.matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, z],
        ]).T

        p = xyz_circle * z_matrix
        points = [*points, *list(p.A)]

    # Shape: [[x1, y1, z1], [x2, y2, z2]]
    return np.matrix(points)


def plot_axis(lengths=[1, 1, 1]):
    line = np.linspace(-1, 1, 10)
    zeros = np.zeros(10)
    x_length, y_length, z_length = lengths

    ax.scatter(line * x_length, zeros, zeros, c='r', marker='o')
    ax.scatter(zeros, line * y_length, zeros, c='g', marker='o')
    ax.scatter(zeros, zeros, line * z_length, c='b', marker='o')

def rotate(matrix, rotation):
    r = R.from_rotvec(rotation)
    return matrix @ r.as_matrix()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cylinder = create_cylinder(1, 5)

# for (x, y, z) in cylinder.A:
#     ax.scatter(x, y, z, c='b', marker='o')


xs, ys, zs = rotate(cylinder.A, [np.pi/4, np.pi/4, 0]).T

ax.scatter(xs, ys, zs, c='r', marker='o')
ax.plot_trisurf(xs, ys, zs)

plot_axis([2, 2, 2])

plt.show()
