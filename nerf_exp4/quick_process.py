import numpy as np 
import copy 

if __name__ == "__main__":
    data = np.load('././dozer2_pi_500.npz')
    images_lb = data['images_lb']
    images_ub = data['images_ub']
    images_noptb = data['images_noptb']
    images_lb_copy = np.minimum(images_lb, images_ub)
    images_ub_copy = np.maximum(images_lb, images_ub)

    np.savez('./dozer2_pi_500.npz', images_lb = images_lb_copy, images_ub = images_ub_copy, images_noptb = images_noptb)
