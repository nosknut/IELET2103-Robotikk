import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_cylinder_surface(coordinates, radius, height, num_points=100):
    phi = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(0, height, num_points)
    Phi, Z = np.meshgrid(phi, z)
    
    X = coordinates[0] + radius * np.cos(Phi)
    Y = coordinates[1] + radius * np.sin(Phi)
    
    cylinder_points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    return cylinder_points

def rotate_points(points, rotation_matrix):
    return np.dot(points, rotation_matrix.T)

# Example cylinder parameters
cylinder_coordinates = np.array([0, 0, 0])  # Center of the cylinder
cylinder_radius = 1.0
cylinder_height = 5.0

cylinder_points = generate_cylinder_surface(cylinder_coordinates, cylinder_radius, cylinder_height)

# Define rotation angles (in radians)
angle_x = np.radians(45)
angle_y = np.radians(45)
angle_z = np.radians(45)

# Create rotation matrices for each axis
rotation_matrix_x = np.array([[1, 0, 0],
                              [0, np.cos(angle_x), -np.sin(angle_x)],
                              [0, np.sin(angle_x), np.cos(angle_x)]])

rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                              [0, 1, 0],
                              [-np.sin(angle_y), 0, np.cos(angle_y)]])

rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                              [np.sin(angle_z), np.cos(angle_z), 0],
                              [0, 0, 1]])

# Apply rotations
cylinder_points = rotate_points(cylinder_points, rotation_matrix_x)
cylinder_points = rotate_points(cylinder_points, rotation_matrix_y)
cylinder_points = rotate_points(cylinder_points, rotation_matrix_z)

def to_mesh(matrix):
    print(matrix.shape[0])
    # Reshape points to meshgrid shape
    num_points = int(np.sqrt(matrix.shape[0]))
    print(num_points)
    X = matrix[:, 0].reshape((num_points, num_points))
    Y = matrix[:, 1].reshape((num_points, num_points))
    Z = matrix[:, 2].reshape((num_points, num_points))
    return [X, Y, Z]


X, Y, Z = to_mesh(cylinder_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, color='b', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rotated 3D Cylinder Surface')

plt.show()
