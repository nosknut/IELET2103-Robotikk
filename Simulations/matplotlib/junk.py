import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate_mesh(points, rotation):
    xs, ys, zs = points
    rx, ry, rz = rotation
    rotation_matrix = np.array([[cos(a), -sin(a)], [sin(a), cos(a)]])
    x, y = zip(*[(x,y) @ rotation_matrix for x,y in zip(x,y)])

def plot_cylinder(pos, radius, height):
    x, y, z = pos
    z_axis = np.linspace(z, z + height, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z_axis)
    x_grid = radius * np.cos(theta_grid) + x
    y_grid = radius * np.sin(theta_grid) + y
    print(x_grid)
    print(y_grid)
    print(z_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_cylinder([1, 1, 1], 4, 5)

plt.show()
