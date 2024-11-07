import numpy as np 

if __name__ == "__main__":
    depth_list = np.random.uniform(0,5,(10,))
    opacity_list = np.random.uniform(0,5,(10,))

    disturbance_depth = np.random.uniform(0,1,(10,))
    disturbance_opacity = np.random.uniform(0,1,(10,))

    depth_upper = depth_list+disturbance_depth
    depth_lower = depth_list-disturbance_depth

    opacity_upper = opacity_list+disturbance_opacity
    opacity_lower = opacity_list-disturbance_opacity

    indices = np.argsort(depth_list)
    print(">>>>>>>> Sorted")
    print(depth_list[indices])

    reorder_nodisturbance = np.zeros((10,10))
    for i in range(10):
        reorder_nodisturbance[i,indices[i]] = 1
    
    print(reorder_nodisturbance@depth_list)
    