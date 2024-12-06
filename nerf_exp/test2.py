import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation

vector = [0,0,1,1]

roll = 0
pitch = 0
yaw = 0 

roll_perturb = np.linspace(-0.5,0.5,21)
pitch_perturb = np.linspace(-0.5,0.5,21)
yaw_perturb = np.linspace(-0.5,0.5,21)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], color='green', linewidth=2, label='Connection Line')

for i in range(20):
    for j in range(20):
        trans = np.zeros((3,4))
        R = Rotation.from_euler('xyz',[roll+roll_perturb[j], pitch+pitch_perturb[i], yaw]).as_matrix()
        T = [0,0,0]
        trans[:3,:3] = R 
        trans[:3,3] = T 
        res = trans@vector 
        ax.plot([0, res[0]], [0, res[1]], [0, res[2]], color='green', linewidth=2, label='Connection Line')

plt.show()