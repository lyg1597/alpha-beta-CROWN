import numpy as np 

if __name__ == "__main__":
    A_l = np.array([
        [0.3158, 0.3203],
        [0.3203, 0.3500]
    ])

    A_u = np.array([
        [0.3196 , 0.3226],
        [0.3226 , 0.3508]
    ])

    A_tmp_min = np.zeros(A_l.shape) +1e10
    A_tmp_max = np.zeros(A_l.shape) -1e10
    for i in range(100):
        A_tmp = np.linalg.inv(np.random.uniform(A_l, A_u))
        A_tmp_min = np.minimum(A_tmp_min, A_tmp)
        A_tmp_max = np.maximum(A_tmp_max, A_tmp)
    print(A_tmp_min, A_tmp_max)