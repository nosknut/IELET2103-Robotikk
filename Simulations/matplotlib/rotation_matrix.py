import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def plot_axis():
    line = np.linspace(-1, 1, 10)
    zeros = np.zeros(10)

    ax.scatter(line, zeros, zeros, c='r', marker='o')
    ax.scatter(zeros, line, zeros, c='g', marker='o')
    ax.scatter(zeros, zeros, line, c='b', marker='o')

r = R.from_rotvec(np.pi/4 * np.array([1, 1, 1]))

m1 = np.matrix([
    [0, 0, 0],
    [1, 0, 0], 
    ])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for (x, y, z) in (m1 @ r.as_matrix()).A:
    ax.scatter(x, y, z, c='black', marker='o', s=50)

plot_axis()

plt.show()


