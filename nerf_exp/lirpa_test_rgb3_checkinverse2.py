import numpy as np 
np.set_printoptions(precision=10)  
x0 = np.eye(2)
x0inv = np.linalg.inv(x0)
xlb = np.array([
    [0.8, -0.2,],
    [-0.2, 0.8]
])
xub = np.array([
    [1.2, 0.2,],
    [0.2, 1.2]
])
dx_max = np.array([
    [0.2, 0.2], 
    [0.2, 0.2]
])
eps_max = np.linalg.norm(dx_max@x0inv)
T = np.linalg.norm(x0inv)*eps_max**2/(1-eps_max)
xll = -np.array([[
    [T, T/np.sqrt(2)],
    [T/np.sqrt(2), T],
]])+(x0inv-x0inv@dx_max@x0inv)
xuu = np.array([[
    [T, T/np.sqrt(2)],
    [T/np.sqrt(2), T],
]])+(x0inv+x0inv@dx_max@x0inv)
print(xll, xuu)
xl_real = np.ones(xll.shape)*np.inf
xu_real = -np.ones(xuu.shape)*np.inf
for i in range(1000):
    # x = np.array([
    #     [0.9,-0.1], 
    #     [-0.1, 1]
    # ])
    x = np.random.uniform(xlb, xub)
    res = np.linalg.inv(x)
    xl_real = np.minimum(res, xl_real)
    xu_real = np.maximum(res, xu_real)
    dx = x-x0 
    eps = np.linalg.norm(dx@x0inv)
    T = np.linalg.norm(x0inv)*eps**2/(1-eps)
    xl = -np.array([[
        [T, T/np.sqrt(2)],
        [T/np.sqrt(2), T],
    ]])+(x0inv-x0inv@dx@x0inv)
    xu = np.array([[
        [T, T/np.sqrt(2)],
        [T/np.sqrt(2), T],
    ]])+(x0inv-x0inv@dx@x0inv)
    # eig0,_ = np.linalg.eig(x0-dx@x0inv)
    # print(eig0)
    # if np.linalg.norm(dx@x0inv)>=1:
    #     print("violated assumption")
    # xl = x0inv-x0inv@dx@x0inv 
    # xu = x0inv-x0inv@dx@x0inv+x0inv@dx@x0inv@dx@x0inv
    if np.any(res<xl) or np.any(res>xu):
        print("violated")
        print(x)
        print(res) 
        print(xl)
        print(xu)
        eig1,_ = np.linalg.eig(xu-res)
        print(eig1)
        eig2,_ = np.linalg.eig(xl-res)
        print(eig2)

    if np.any(res<xll) or np.any(res>xuu):
        print("violated Global bound ")
        print(x)
        print(res) 
        print(xl)
        print(xu)
        eig1,_ = np.linalg.eig(xu-res)
        print(eig1)
        eig2,_ = np.linalg.eig(xl-res)
        print(eig2)
print(xl_real, xu_real)