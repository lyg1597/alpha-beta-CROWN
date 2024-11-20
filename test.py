import scipy.linalg
import numpy as np 
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    A = np.array([[0,-1],[1,-1]])

    t = np.linspace(0,10,1000)

    At = np.expand_dims(A, axis=0)*np.expand_dims(t, axis=(1,2))

    res = scipy.linalg.expm(At)

    x0 = np.random.uniform(-1,1,(100,2,1)) 

    for i in range(x0.shape[0]):
        xt = res@x0[i,:] 

        plt.plot(xt[:,0], xt[:,1])

    plt.show()