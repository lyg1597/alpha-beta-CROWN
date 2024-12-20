import torch 
import numpy as np 
from scipy.spatial.transform import Rotation
from simple_model import RasterizationModelRGBManual_notile, DepthModel
import matplotlib.pyplot as plt 
import pyvista as pv

N = 3

def visualize_scene(means: np.ndarray, covs: np.ndarray, colors: np.ndarray, opacities: np.ndarray):
    # Create a plotter
    plotter = pv.Plotter()

    # Create three axes lines starting at (0,0,0)
    x_line = pv.Line((0,0,0), (10,0,0))
    y_line = pv.Line((0,0,0), (0,10,0))
    z_line = pv.Line((0,0,0), (0,0,10))

    # Add the axes lines to the plotter with different colors
    plotter.add_mesh(x_line, color='red', line_width=5)
    plotter.add_mesh(y_line, color='green', line_width=5)
    plotter.add_mesh(z_line, color='blue', line_width=5)

    # Define the box boundaries:
    # bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Cube(bounds=(-2, 2, -2, 2, 10, 11))

    # Add the box as a wireframe
    plotter.add_mesh(box, color='white', style='wireframe', line_width=2)

    for i in range(len(covs)):
        mean = means.squeeze()[i]
        cov = covs.squeeze()[i]
        opacity = opacities.squeeze()[i]
        color = colors.squeeze()[i]

        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors by largest eigenvalue first
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Choose your "confidence interval" (scaling factor):
        # For example, 1-sigma ellipsoid:
        scales = np.sqrt(eigvals)  # Semi-axis lengths

        # Create a unit sphere
        sphere = pv.Sphere(radius=1.0, center=(0,0,0), phi_resolution=30, theta_resolution=30)

        transform = np.eye(4)
        transform[:3, :3] = eigvecs @ np.diag(scales)
        transform[:3, 3] = mean

        # Apply the transformation
        ellipsoid = sphere.transform(transform)

        # Add the ellipsoid to the plot
        plotter.add_mesh(ellipsoid, color=color, opacity=0.5)

    plotter.show()

if __name__ == "__main__":
    w = 20
    h = 20
    # A straight up camera matrix
    camera_pose = torch.Tensor(np.array([[
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]])).to('cuda')
    # means of three gaussian
    # means = np.random.uniform([0,0,10],[5,5,15],(1,N,2))
    means = np.array([
        [-1,0, 10.5],
        [0,0, 10.5],
        [1,0, 10.5]
    ])
    # Orientations of three gaussian
    # rpys = np.random.uniform([-np.pi/2,-np.pi/2,-np.pi/2], [np.pi/2,np.pi/2,np.pi/2], (1,N,3))
    rpys = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ])
    # Scales of three gaussian, all scales between -1-0
    # scales = np.random.uniform(-1, -0.2, (1,N,3))
    scales = np.array([
        [-0.2,-0.2,-0.2],
        [-0.2,-0.2,-0.2],
        [-0.2,-0.2,-0.2]
    ])
    # Setup Opacities of three gaussian, all between 0.5-0.8 (relatively opaque)
    # opacities = np.random.uniform(0.5, 0.8, (1,N,1))
    opacities = np.array([
        [0.6],
        [0.6],
        [0.6]
    ])
    # Setup colors of three gaussian, basically r,g,b if N=3 and r,g,b,y,p,o if N=6
    if N==3:
        colors = torch.Tensor(np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])).to('cuda')
    elif N==6:
        colors = torch.Tensor(np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [0,1,1],
            [1,0,1]
        ])).to('cuda')
    else:
        raise ValueError
    
    # quats = Rotation.from_euler('xyz', rpys).as_quat(scalar_first=True) # w,x,y,z
    quats = Rotation.from_euler('xyz', rpys).as_quat() # x,y,z,w
    quats = np.hstack((quats[:,3:4], quats[:,0:1], quats[:,1:2], quats[:,2:3]))
    
    matrices = Rotation.from_euler('xyz', rpys).as_matrix()
    # quats = np.expand_dims(quats, axis=0)

    covs = []
    for i in range(scales.shape[0]):
        scale = np.exp(scales[i])
        scale_matrix = np.diag(scale)
        M = matrices[i]@scale_matrix
        cov = M@M.T
        covs.append(cov)
    covs = np.array(covs)

    visualize_scene(means, covs, colors.cpu().numpy(), opacities)

    data_pack = {
        'opacities': torch.FloatTensor(opacities),
        'means': torch.FloatTensor(means),
        'scales':torch.FloatTensor(scales),
        'quats':torch.FloatTensor(quats)
    }

    model_alpha = RasterizationModelRGBManual_notile(
        data_pack=data_pack,
        fx=w*2,
        fy=h*2,
        width=w,
        height=h,
    )
    print("###### Model Alpha")

    model_depth = DepthModel(model_alpha)
    print("###### Model Depth")

    res_alpha = model_alpha(camera_pose)
    print("###### Alpha")
    res_depth = model_depth(camera_pose)
    print("###### Depth")
    depth_order = torch.argsort(res_depth, dim=1).squeeze()
    sorted_alpha = res_alpha[0,:,depth_order,:]
    sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    sorted_color = colors[depth_order,:]
    rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
    rgb_color = rgb_color.reshape(w, h, -1)[:,:,:3]
    rgb_color = rgb_color.detach().cpu().numpy()
    plt.figure(3)
    plt.imshow(rgb_color)
    plt.show()
    