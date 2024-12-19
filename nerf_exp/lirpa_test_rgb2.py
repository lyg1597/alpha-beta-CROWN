import torch 
import numpy as np 
from scipy.spatial.transform import Rotation
from rasterize_model import RasterizationModelRGBManual_notile

N = 3

if __name__ == "__main__":
    # means of three gaussian
    means = np.random.uniform([0,0,10],[5,5,15],(1,N,2))

    # Orientations of three gaussian
    rpys = np.random.uniform([-np.pi/2,-np.pi/2,-np.pi/2], [np.pi/2,np.pi/2,np.pi/2], (1,N,3))
    quats = Rotation.from_euler('xyz', rpys).as_quat() # x,y,z,w

    # Scales of three gaussian, all scales between -1-0
    scales = np.random.uniform(-1, -0.2, (1,N,3))

    # Setup Opacities of three gaussian, all between 0.5-0.8 (relatively opaque)
    opacities = np.random.uniform(0.5, 0.8, (1,N,1))

    # Setup colors of three gaussian, basically r,g,b if N=3 and r,g,b,y,p,o if N=6
    if N==3:
        colors = np.expand_dims(np.array([
            [255,0,0],
            [0,255,0],
            [0,0,255]
        ]), axis=0)
    elif N==6:
        colors = np.expand_dims(np.array([
            [255,0,0],
            [0,255,0],
            [0,0,255]
            [255,255,0],
            [0,255,255],
            [255,0,255]
        ]), axis=0)
    else:
        raise ValueError
    
    # A straight up camera matrix
    camera_pose = np.array([
        [1,0,0,2.5],
        [0,1,0,2.5],
        [0,0,1,0],
        [0,0,0,1]
    ])

    data_pack = {
        'opacities': torch.FloatTensor(opacities),
        'means': torch.FloatTensor(means),
        'scales':torch.FloatTensor(scales),
        'quats':torch.FloatTensor(quats)
    }

