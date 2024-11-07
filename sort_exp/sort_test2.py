import numpy as np 
import matplotlib.pyplot as plt 
import copy 

def compute_order(vec):
    tmp_vec = copy.deepcopy(vec)
    order_array = np.arange(0,10,1)
    for i in range(len(vec)-1):
        for j in range(len(vec)-1-i):
            if tmp_vec[j+1,:,1]<tmp_vec[j,:,0]:
                tmp = order_array[j]
                order_array[j] = order_array[j+1]
                order_array[j+1] = tmp

                tmp = copy.deepcopy(tmp_vec[j,:,:])
                tmp_vec[j,:,:] = tmp_vec[j+1,:,:]
                tmp_vec[j+1,:,:] = tmp
            elif tmp_vec[j+1,:,1]<tmp_vec[j,:,1]:
                tmp = order_array[j]
                order_array[j] = order_array[j+1]
                order_array[j+1] = tmp 

                tmp = copy.deepcopy(tmp_vec[j,:,:])
                tmp_vec[j,:,:] = tmp_vec[j+1,:,:]
                tmp_vec[j+1,:,:] = tmp
    print(order_array)
    # return None 
    order_matrix = np.zeros((10,10,2))
    for i in range(len(vec)-1):
        if tmp_vec[i,0,1]<tmp_vec[i+1,0,0]:
            order_matrix[i,order_array[i],0] = 1
            order_matrix[i,order_array[i],1] = 1
        else:
            order_matrix[i,order_array[i],0] = 0
            order_matrix[i,order_array[i],1] = 1
            for j in range(i+1, len(vec)):
                if tmp_vec[i,0,1]>=tmp_vec[j,0,0]:
                    order_matrix[i,order_array[j],0] = 0
                    order_matrix[i,order_array[j],1] = 1
                    order_matrix[j,order_array[i],0] = 0
                    order_matrix[j,order_array[i],1] = 1
                    order_matrix[i+1,order_array[j],0] = 0
                    order_matrix[i+1,order_array[j],1] = 1
    print(order_matrix)
    return order_matrix

def apply_reorder1(vec, reorder):
    res = np.zeros(vec.shape)
    for i in range(reorder.shape[0]):
        for j in range(reorder.shape[1]):
            res[i,0,0] += min(vec[j,0,0]*reorder[i,j,0], vec[j,0,0]*reorder[i,j,1],vec[j,0,1]*reorder[i,j,0], vec[j,0,1]*reorder[i,j,1])
            res[i,0,1] += max(vec[j,0,0]*reorder[i,j,0], vec[j,0,0]*reorder[i,j,1],vec[j,0,1]*reorder[i,j,0], vec[j,0,1]*reorder[i,j,1])
    return res 

def apply_reorder2(vec, reorder):
    res = np.zeros(vec.shape)
    res[:,0,0] = np.double('inf')
    for i in range(reorder.shape[0]):
        # max_min = np.double('inf')
        for j in range(reorder.shape[1]):
            if reorder[i,j,1]==1:
                res[i,0,0] = min(res[i,0,0], vec[j,0,0])
                # max_min = min(max_min, vec[i,0,0])
                res[i,0,1] = max(res[i,0,1], vec[j,0,1])
    return res 

if __name__ == "__main__":
    # pos_x = np.expand_dims(np.random.uniform(0,10,10),1)
    # pos_y = np.expand_dims(np.random.uniform(0,10,10),1)
    pos_x = np.array([
        [1.93514307],
        [2.70213485],
        [8.78341511],
        [9.56918586],
        [1.00138134],
        [7.60500773],
        [9.03599574],
        [4.2826591 ],
        [2.3720792 ],
        [0.0292233 ],
    ])
    pos_y = np.array([
        [4.17699007],
        [7.31911928],
        [2.64857697],
        [5.35833646],
        [5.8297098 ],
        [2.25268658],
        [6.497207  ],
        [6.99482564],
        [1.95171219],
        [3.30844635],
    ])

    pos = np.concatenate((pos_x, pos_y), axis=1)

    print(pos_x)
    print(pos_y)

    plt.figure(0)

    plt.plot(pos_x, pos_y, 'r*')
    # plt.show()

    # x_low = 1
    # x_high = 2

    x_center = 2
    x_radius = 1.0

    plt.plot(x_center, 0, 'b*')

    dist_center = np.linalg.norm(np.array([x_center,0])-pos,ord=1, axis=1)    
    dist_high = pos_y + np.maximum(np.abs(pos_x-(x_center+x_radius)), np.abs(pos_x-(x_center-x_radius)))
    dist_low = pos_y + np.maximum(np.sign(((x_center+x_radius)-pos_x)*((x_center-x_radius)-pos_x))*np.minimum(np.abs(pos_x-(x_center+x_radius)), np.abs(pos_x-1.5)),0)

    dist_high = np.expand_dims(dist_high, axis=2)
    dist_low = np.expand_dims(dist_low, axis=2)
    dist = np.concatenate((dist_low, dist_high), axis=2)

    print(dist)

    plt.figure(1)
    plt.plot(dist_center)
    plt.plot(dist_high[:,0,0])
    plt.plot(dist_low[:,0,0])

    opacities = np.array([0.77353248, 0.0596479 , 0.118955  , 0.75224626, 0.15769205,
       0.58342989, 0.88040631, 0.24178431, 0.3251487 , 0.50760633])
    opacities = np.expand_dims(opacities, axis=(1,2))
    opacities = np.concatenate((opacities, opacities), axis=2)
    print(opacities.shape)

    colors_center = np.array([3.4021099 , 1.15162148, 2.77999899, 3.9836582 , 4.24928171,
       0.44861543, 2.16127997, 1.9492195 , 1.57719062, 2.67486213])
    colors_center = np.expand_dims(colors_center, axis=(1,2))
    radius = np.random.uniform(0,1,(10,1,1))
    colors_low = colors_center-radius 
    colors_high = colors_center+radius
    colors = np.concatenate((colors_low, colors_high), axis=2)
    print(colors.shape)

    reorder = np.zeros((10,10,2))
    reorder_center = np.zeros((10,10,2)) 
    order_center = np.argsort(dist_center)
    # for i in range(len(order_center)):
    #     reorder_center[i, reorder_center]

    reorder_res = compute_order(dist)

    opacities_reorder = apply_reorder1(opacities, reorder_res)
    colors_reorder = apply_reorder1(colors, reorder_res)

    print(opacities_reorder[:,0,0],opacities_reorder[:,0,1])
    print(colors_reorder[:,0,0],colors_reorder[:,0,1])

    opacities_reorder = apply_reorder2(opacities, reorder_res)
    colors_reorder = apply_reorder2(colors, reorder_res)

    print(opacities_reorder[:,0,0])
    print(opacities_reorder[:,0,1])
    print(colors_reorder[:,0,0])
    print(colors_reorder[:,0,1])


    # plt.figure(0)
    # for x in range(10):
    #     order = np.argsort(np.linalg.norm(np.array([x,0])-pos,ord=1, axis=1))
    #     plt.plot(order, label=f"{x}")

    # plt.legend()

    # plt.figure(1)
    # plt.plot(pos_x, pos_y, 'r*')
    plt.show()        
    
