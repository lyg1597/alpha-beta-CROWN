import numpy as np 
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    data = np.load('dozer2_pi_1000_tight.npz')
    images_lb = data['images_lb']
    images_ub = data['images_ub']
    camera_poses = data['images_noptb']

    for i in range(100):
        print(camera_poses[i])
        plt.figure(0)
        plt.imshow(images_lb[i])
        plt.figure(1)
        plt.imshow(images_ub[i])
        plt.show()