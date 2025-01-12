import torch 
import numpy as np 
from scipy.spatial.transform import Rotation
from simple_model2_inversetest import AlphaModel, InverseModelLb, InverseModelUb, AlphaModel2
import matplotlib.pyplot as plt 
import pyvista as pv
from typing import List, Dict
from collections import defaultdict
import itertools
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

def reorg_bounds(bounds, pivot):
    pivot_val = bounds[pivot]
    # res = [pivot_val]
    bounds_left = []
    bounds_right = []
    for i in range(len(bounds)):
        if i!=pivot:
            val = bounds[i]
            if val[1] <= pivot_val[0]:
                # res = [val]+res
                bounds_left.append(val) 
            elif pivot_val[1] <= val[0]:
                bounds_right.append(val)
            elif val[0] < pivot_val[0]:
                bounds_left.append(val)
            else:
                bounds_right.append(val)
    return bounds_left, bounds_right

def sort_bounds(bounds:List[List[float]]):
    if len(bounds) == 0:
        return bounds
    elif len(bounds) == 1:
        return bounds
    else:
        pivot = int(len(bounds)/2)
        bounds_left, bounds_right = reorg_bounds(bounds, pivot)
        sort_left = sort_bounds(bounds_left)
        sort_right = sort_bounds(bounds_right)
        return sort_left + [bounds[pivot]] + sort_right

def get_set_order(sorted_bounds):
    res_list = []
    for i in range(len(sorted_bounds)):
        bins = []
        ref_bound = sorted_bounds[i]
        for j in range(len(sorted_bounds)):
            bound = sorted_bounds[j]
            if ref_bound[0]<=bound[1]<=ref_bound[1] or \
                ref_bound[0]<=bound[0]<=ref_bound[1] or \
                bound[0]<=ref_bound[0]<=bound[1] or \
                bound[0]<=ref_bound[1]<=bound[1]:
                bins.append(bound[2])
        res_list.append(bins)
    return res_list

def get_elem_before_linear(ptb: PerturbationLpNorm, A: defaultdict, model:BoundedModule):
    x_L = ptb.x_L
    x_U = ptb.x_U
    x_bounds = torch.cat((x_L, x_U), dim=0)
    x_bounds_list = x_bounds.transpose(0,1).detach().cpu().numpy().tolist()
    all_combins = list(itertools.product(*x_bounds_list))

    lA: torch.Tensor = A[model.output_name[0]][model.input_name[0]]['lA']
    uA: torch.Tensor = A[model.output_name[0]][model.input_name[0]]['uA']
    lbias: torch.Tensor = A[model.output_name[0]][model.input_name[0]]['lbias']
    ubias: torch.Tensor = A[model.output_name[0]][model.input_name[0]]['ubias']
    
    concrete_bound = {}
    possible_bound = {}
    for i in range(lA.shape[1]):
        concrete_bound[i] = []
        possible_bound[i] = []
        for j in range(lA.shape[1]):
            if i==j:
                continue
            smaller_than = False 
            greater_than = False 
            for k in range(len(all_combins)):
                x = torch.Tensor(all_combins[k]).to(lA.device)
                fxl = lA[0,j,:]@x+lbias[0,j]
                fxu = uA[0,j,:]@x+ubias[0,j]
                fxl_ref = lA[0,i,:]@x+lbias[0,i]
                fxu_ref = uA[0,i,:]@x+ubias[0,i]
                if fxu<=fxl_ref:
                    smaller_than = smaller_than or True 
                elif fxl>=fxu_ref:
                    greater_than = greater_than or True
            if smaller_than and not greater_than:
                concrete_bound[i].append(j)
                possible_bound[i].append(j)
            elif smaller_than and greater_than:
                possible_bound[i].append(j)
            elif not smaller_than and not greater_than:
                possible_bound[i].append(j)
    return concrete_bound, possible_bound

def get_elem_before(sorted_bounds):
    concrete_bound = {}
    possible_bound = {}
    for i in range(len(sorted_bounds)):
        ref_bound = sorted_bounds[i]
        concrete_bound[ref_bound[2]] = []
        possible_bound[ref_bound[2]] = []
        for j in range(len(sorted_bounds)):
            bound = sorted_bounds[j]
            # First check concrete bound
            # If lower bound of ref is greater than upper bound of ref, then bound is guaranteed before ref_bound 
            if ref_bound[0]>bound[1] and ref_bound[2]!=bound[2]:
                # Add bound to both concrete and reference bound 
                concrete_bound[ref_bound[2]].append(bound[2])
                possible_bound[ref_bound[2]].append(bound[2])
            # Check if over-lapping bound, if yes, only add to possible bound              
            elif (ref_bound[0]<=bound[1]<=ref_bound[1] or \
                ref_bound[0]<=bound[0]<=ref_bound[1] or \
                bound[0]<=ref_bound[0]<=bound[1] or \
                bound[0]<=ref_bound[1]<=bound[1]) and ref_bound[2]!=bound[2]:
                possible_bound[ref_bound[2]].append(bound[2])
    return concrete_bound, possible_bound
        
def bounds_union(b1, b2):
    '''
    b1: 2*m
    b2: 2*m
    '''
    b_out = torch.zeros(b1.shape)
    b_out[0,:] = torch.min(torch.stack((b1[0,:],b2[0,:]), dim=0), dim=0).values
    b_out[1,:] = torch.max(torch.stack((b1[1,:],b2[1,:]), dim=0), dim=0).values
    return b_out 

def computeT(concrete_before: Dict, possible_before: Dict, bounds_alpha: torch.Tensor):
    T = torch.ones(bounds_alpha.shape).to(bounds_alpha.device)
    for i in range(bounds_alpha.shape[2]):
        # Compute lb, using possible bounds, use upper bound of alpha 
        for j in range(len(possible_before[i])):
            T[0, :, i, :] = T[0, :, i, :]*(1-bounds_alpha[1,:,possible_before[i][j],:])
        # Compute ub, using concrete bounds, use lower bound of alpha 
        for j in range(len(concrete_before[i])):
            T[1, :, i, :] = T[1, :, i, :]*(1-bounds_alpha[0,:,concrete_before[i][j],:])
    return T 

def apply_set_order(set_order: List, bounds: torch.Tensor):
    '''
    set_order: List with length N
    bounds: 2*256*N 
    '''
    sorted_bounds = torch.zeros(bounds.shape).to(bounds.device)
    for i in range(len(set_order)):
        for j in range(len(set_order[i])):
            if j==0:
                sorted_bounds[:,:,i] = bounds[:,:,set_order[i][j]]
            else:
                sorted_bounds[:,:,i] = bounds_union(sorted_bounds[:,:,i], bounds[:,:,set_order[i][j]])
    return sorted_bounds

def compute_sortedT(sorted_alpha: List):
    # Initialize lb, ub for the \prod (1-\alpha) term 
    alpha_lb = sorted_alpha[0,...]
    # alpha
    alpha_ub = sorted_alpha[1,...]

    T_lb = torch.cat([torch.ones_like(alpha_lb[:,:1]), 1-alpha_ub[:,:-1]], dim=1).cumprod(dim=1)
    T_ub = torch.cat([torch.ones_like(alpha_ub[:,:1]), 1-alpha_lb[:,:-1]], dim=1).cumprod(dim=1)
    T_bound = torch.stack((T_lb, T_ub), dim=0)
    return T_bound

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
        scales = np.sqrt(eigvals)*3  # Semi-axis lengths

        # Create a unit sphere
        sphere = pv.Sphere(radius=1.0, center=(0,0,0), phi_resolution=30, theta_resolution=30)

        transform = np.eye(4)
        transform[:3, :3] = eigvecs @ np.diag(scales)
        transform[:3, 3] = mean

        # Apply the transformation
        ellipsoid = sphere.transform(transform)

        # Add the ellipsoid to the plot
        plotter.add_mesh(ellipsoid, color=color, opacity=0.5)

    # Camera parameters
    width = 1.0      # width = height
    focal_length = 2 * width
    z_dist = focal_length
    half_w = width / 2

    # Define the points of the camera pyramid
    apex = (0.0, 0.0, 0.0)  # camera center (at origin)
    p1 = (-half_w, -half_w, z_dist)
    p2 = ( half_w, -half_w, z_dist)
    p3 = ( half_w,  half_w, z_dist)
    p4 = (-half_w,  half_w, z_dist)

    # Combine all points into an array
    points = np.array([apex, p1, p2, p3, p4])

    # Define the faces of the pyramid
    # PyVista’s PolyData faces format: [npts, pt0, pt1, pt2, ...]
    # Apex is point 0, p1=1, p2=2, p3=3, p4=4
    faces = np.hstack([
        [3, 0, 1, 2],  # triangle apex-p1-p2
        [3, 0, 2, 3],  # triangle apex-p2-p3
        [3, 0, 3, 4],  # triangle apex-p3-p4
        [3, 0, 4, 1]   # triangle apex-p4-p1
    ])

    # Create a polydata for the pyramid
    camera_pyramid = pv.PolyData(points, faces)

    # Add the pyramid to the plot
    # You can change the color and style as you wish
    plotter.add_mesh(camera_pyramid, color='magenta', style='wireframe', line_width=2)


    plotter.show()

if __name__ == "__main__":
    eps = 0.05
    w = 20
    h = 20
    # A straight up camera matrix
    # camera_pose = torch.Tensor(np.array([[
    #     [1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,1,0],
    #     [0,0,0,1]
    # ]])).to('cuda')
    camera_pose = torch.Tensor(np.array([[
        0,0,0,0,0,0
    ]])).to('cuda')
    # means of three gaussian
    # means = np.random.uniform([0,0,10],[5,5,15],(1,N,2))
    means = np.array([
        [-2,0, 10],
        [0,0, 10.5],
        [2,0, 11]
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
        [-0.4,-0.2,-0.4],
        [-0.2,-0.2,-0.4],
        [-0.2,-0.4,-0.4]
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
            [1.0,0,0],
            [0,1.0,0],
            [0,0,1.0]
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
    
    data_pack = {
        'opacities': torch.FloatTensor(opacities),
        'means': torch.FloatTensor(means),
        'scales':torch.FloatTensor(scales),
        'quats':torch.FloatTensor(quats)
    }

    model_alpha = AlphaModel(
        data_pack=data_pack,
        fx=w*2,
        fy=h*2,
        width=w,
        height=h,
    )
    A_lb = torch.Tensor([[
        [96794.266, -718.9259],
        [-718.9259,  96794.3]
    ]])
    A_ub = torch.Tensor([[
        [142558.67, 718.92676],
        [718.92676, 142558.69]
    ]])
    A0 = ((A_lb+A_ub)/2).to(model_alpha.device)
    # J0 = torch.Tensor(np.array([
    #     [420,0,0],[0,420,0]
    # ])).to(model_alpha.device)
    # cov = model_alpha.cov_world[1]
    # A0 = J0@cov@J0.T
    # A0 = A0[None]
    print("###### Model Alpha")

    model_inverse_lb = InverseModelLb(A0)
    model_inverse_ub = InverseModelUb(A0)
    # torch.onnx.export(model_alpha, camera_pose, 'model.onnx') 

    ##################### Compute Bounds #####################
    print(">>>>>> Starting PerturbationLpNorm")
    # ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    ptb_A = PerturbationLpNorm(
        norm=np.inf, 
        x_L=A_lb.to(model_inverse_lb.device),
        x_U=A_ub.to(model_inverse_lb.device),
    )
    print(">>>>>> Starting BoundedTensor")
    my_input = BoundedTensor(A0, ptb_A)
    print(">>>>>> Starting Bounded Module")
    model_inverselb_bounded = BoundedModule(model_inverse_lb, A0, device=model_inverse_lb.device, bound_opts={'conv_mode': 'matrix'})
    prediction = model_inverselb_bounded(my_input)
    model_inverselb_bounded.visualize('inverse_lb')
    print(">>>>>> Starting Compute Bound")
    lb_inverselb, ub_inverselb = model_inverselb_bounded.compute_bounds(x=(my_input, ), method='ibp')
    print(">>>>>> Starting Bounded Module")
    model_inverseub_bounded = BoundedModule(model_inverse_ub, A0, device=model_inverse_ub.device, bound_opts={'conv_mode': 'matrix'})
    prediction = model_inverseub_bounded(my_input)
    model_inverseub_bounded.visualize('inverse_ub')
    print(">>>>>> Starting Compute Bound")
    lb_inverseub, ub_inverseub = model_inverseub_bounded.compute_bounds(x=(my_input, ), method='ibp')
    
    print(lb_inverselb, ub_inverseub)

    lb_inverse = lb_inverselb.detach().cpu().numpy()
    ub_inverse = ub_inverseub.detach().cpu().numpy()
    lb_inverseub = lb_inverseub.detach().cpu().numpy()
    ub_inverseub = ub_inverseub.detach().cpu().numpy()
    for i in range(100):
        lambda_val = torch.rand((1,2,2))
        inp = lambda_val*A_lb+(1-lambda_val)*A_ub 
        if torch.any(inp<A_lb) or torch.any(A_ub<inp):
            print("wrong input bound")
        res = model_inverse_ub(inp.to(model_inverse_ub.device))
        res = res.detach().cpu().numpy()
        if np.any(res<lb_inverseub) or np.any(ub_inverseub<res):
            print("bound violate 1")
        res2 = torch.inverse(inp)
        res2 = res2.detach().cpu().numpy()
        if np.any(res<lb_inverse) or np.any(ub_inverse < res2) :
            print("bound violate 2")

    
    # plt.figure(7)
    # plt.imshow(diff_compemp_lb)
    # plt.title('diff_comp_emp_lb')
    # plt.figure(8)
    # plt.imshow(diff_compemp_ub)
    # plt.title('diff_comp_emp_ub')
    # plt.show()