import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def plot_axis():
    line = np.linspace(-1, 1, 10)
    zeros = np.zeros(10)

    ax.scatter(line, zeros, zeros, c='r', marker='o')
    ax.scatter(zeros, line, zeros, c='g', marker='o')
    ax.scatter(zeros, zeros, line, c='b', marker='o')

def plot_cylinder(pos, radius, height):
    x, y, z = pos
    z_axis = np.linspace(0, height, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z_axis)
    x_grid = radius * np.cos(theta_grid) + x
    y_grid = radius * np.sin(theta_grid) + y
    print(x_grid)
    mat = np.matrix([x_grid, y_grid, z_grid])
    return mat

r = R.from_rotvec(np.pi/4 * np.array([1, 1, 1]))
r.as_matrix()

m1 = np.matrix([
    [0, 0, 0],
    [1, 0, 0], 
    ])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# cylinder = plot_cylinder([0, 0, 0], 4, 5)

for (x, y, z) in (m1 @ r.as_matrix()).A:
    ax.scatter(x, y, z, c='black', marker='o', s=50)

plot_axis()

plt.show()


