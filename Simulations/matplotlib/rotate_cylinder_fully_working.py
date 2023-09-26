import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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

def plot_axis(ax, lengths=[1, 1, 1]):
    x_length, y_length, z_length = lengths
    
    ax.plot([-x_length, x_length], [0, 0], [0, 0], c='r', marker='o')
    ax.plot([0, 0], [-y_length, y_length], [0, 0], c='g', marker='o')
    ax.plot([0, 0], [0, 0], [-z_length, z_length], c='b', marker='o')

def rotate(matrix, rotation):
    r = R.from_rotvec(rotation)
    return matrix @ r.as_matrix()    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cylinder = meshgrid_to_matrix(create_cylinder(1, 6, 20))

rotated = cylinder @ R.from_rotvec([np.pi/2, np.pi/2, np.pi/2]).as_matrix()

ax.plot_surface(*matrix_to_meshgrid(cylinder), color='b', alpha=0.5)
ax.plot_surface(*matrix_to_meshgrid(rotated), color='r', alpha=0.5)

plot_axis(ax, [3, 3, 3])

plt.show()
