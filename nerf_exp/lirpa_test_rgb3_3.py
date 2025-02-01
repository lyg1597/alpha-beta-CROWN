import torch 
import numpy as np 
from scipy.spatial.transform import Rotation
from simple_model2_alphatest2 import AlphaModel, DepthModel
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
        [0.3],
        [0.3],
        [0.3]
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

    model_alpha = AlphaModel(
        data_pack=data_pack,
        fx=w*2,
        fy=h*2,
        width=w,
        height=h,
    )
    print("###### Model Alpha")

    model_depth = DepthModel(model_alpha)
    print("###### Model Depth")

    # res_alpha = model_alpha(camera_pose)
    # print("###### Alpha")
    # res_depth = model_depth(camera_pose)
    # print("###### Depth")
    # depth_order = torch.argsort(res_depth, dim=1).squeeze()
    # sorted_alpha = res_alpha[0,:,depth_order,:]
    # sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    # sorted_color = colors[depth_order,:]
    # rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
    # rgb_color = rgb_color.reshape(w, h, -1)[:,:,:3]
    # rgb_color = rgb_color.detach().cpu().numpy()
    # plt.figure(0)
    # plt.imshow(rgb_color)
    # plt.show()

    ##################### Compute Bounds #####################
    my_input = torch.clone(camera_pose)
    print(">>>>>> Starting Bounded Module")
    model_alpha_bounded = BoundedModule(
        model_alpha, 
        my_input, 
        device=model_alpha.device, 
        bound_opts= {
            'conv_mode': 'matrix',
            'optimize_bound_args': {'iteration': 20},
        }, 
    )
    print(">>>>>> Starting PerturbationLpNorm")
    # ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    ptb_alpha = PerturbationLpNorm(
        norm=np.inf, 
        x_L=torch.Tensor([[-0.0,-0.0,-0.0,-1.0,-1.0,-1.0]]).to(model_alpha.device),
        x_U=torch.Tensor([[0.0,0.0,0.0,1.0,1.0,1.0]]).to(model_alpha.device),
    )
    print(">>>>>> Starting BoundedTensor")
    my_input = BoundedTensor(my_input, ptb_alpha)
    prediction = model_alpha_bounded(my_input)
    model_alpha_bounded.visualize('alpha')
    print(">>>>>> Starting Compute Bound")
    required_A = defaultdict(set)
    required_A[model_alpha_bounded.output_name[0]].add(model_alpha_bounded.input_name[0])
    # lb_alpha, ub_alpha, A_alpha = model_alpha_bounded.compute_bounds(x=(my_input, ), method='crown', return_A=True, needed_A_dict=required_A)
    lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(my_input, ), method='alpha-crown')
    lb_alpha = torch.clip(lb_alpha, min=0)
    ub_alpha = torch.clip(ub_alpha, max=0.99)
    bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)
    
    model_depth = DepthModel(model_alpha)
    my_input = torch.clone(camera_pose)
    print(">>>>>> Starting Bounded Module")
    model_depth_bounded = BoundedModule(model_depth, my_input, device=model_depth.device, bound_opts={'conv_mode': 'matrix'})
    print(">>>>>> Starting PerturbationLpNorm")
    # ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    ptb_depth = PerturbationLpNorm(
        norm=np.inf, 
        x_L=torch.Tensor([[-0.05,-0.05,-0.05,-1.0,-1.0,-1.0]]).to(model_depth.device),
        x_U=torch.Tensor([[0.05,0.05,0.05,1.0,1.0,1.0]]).to(model_depth.device),
    )
    print(">>>>>> Starting BoundedTensor")
    my_input = BoundedTensor(my_input, ptb_depth)
    prediction = model_depth_bounded(my_input)
    required_A = defaultdict(set)
    required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
    lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(my_input, ), method='crown', return_A=True, needed_A_dict=required_A)
    # lb_depth, ub_depth = model_depth_bounded.compute_bounds(x=(my_input, ), method='ibp')

    lb_depth = lb_depth.detach().cpu().numpy()    
    ub_depth = ub_depth.detach().cpu().numpy()    
    bounds_depth = np.vstack((lb_depth, ub_depth)).T
    bounds_depth = bounds_depth.tolist()
    bounds_depth = [elem+[i] for i, elem in enumerate(bounds_depth)]
    sorted_bounds = sort_bounds(bounds_depth)

    # concrete_before, possible_before = get_elem_before(bounds_depth)
    concrete_before, possible_before = get_elem_before_linear(ptb_depth, A_depth, model_depth_bounded)
    print(concrete_before, possible_before)
    res_T = computeT(concrete_before, possible_before, bounds_alpha)
    
    res_2d = colors
    bounds_res_2d = torch.stack((res_2d, res_2d), dim=0)
    bounds_res_2d = bounds_res_2d[:,None]
    tile_color = (res_T*bounds_alpha*bounds_res_2d).sum(dim=2)
    
    set_order = get_set_order(sorted_bounds)
    set_sorted_alpha = apply_set_order(set_order, bounds_alpha)
    set_sorted_T = compute_sortedT(set_sorted_alpha)
    bounds_res_2d = torch.stack((colors, colors), dim=0)
    bounds_res_2d = bounds_res_2d[:,None]
    bounds_alphac = bounds_alpha*bounds_res_2d
    set_sorted_alphac = apply_set_order(set_order, bounds_alphac)
    tile_color_old = (set_sorted_T*bounds_alphac).sum(dim=2)
    diff = torch.abs(tile_color-tile_color_old).sum(dim=2)

    tile_color_lb = tile_color[0,:,:3].reshape((w,h,-1))
    tile_color_lb = tile_color_lb.detach().cpu().numpy()
    tile_color_ub = tile_color[1,:,:3].reshape((w,h,-1))
    tile_color_ub = tile_color_ub.detach().cpu().numpy()
    diff_lb = diff[0,:].reshape((w,h,-1))
    diff_lb = diff_lb.detach().cpu().numpy()
    diff_ub = diff[1,:].reshape((w,h,-1))
    diff_ub = diff_ub.detach().cpu().numpy()

    empirical_lb = np.zeros(tile_color_lb.shape)+1e10
    empirical_ub = np.zeros(tile_color_lb.shape)
    empirical_alpha_lb = np.zeros(lb_alpha.shape)+1e10
    empirical_alpha_ub = np.zeros(ub_alpha.shape)
    lb_alpha = lb_alpha.detach().cpu().numpy()
    ub_alpha = ub_alpha.detach().cpu().numpy()
    for i in range(1000):
        tmp_input = my_input.repeat(1,1)
        delta = torch.zeros((1,6))
        # delta[:,:3,3] = torch.rand((1,3))*eps*2-eps
        delta[:,:3] = torch.rand((1,3))*0.0*2-0.0
        delta[:,3:] = torch.rand((1,3))*1.0*2-1.0
        delta = delta.to(model_depth.device)
        tmp_input = tmp_input+delta 
        res_alpha = model_alpha(tmp_input)
        res_depth = model_depth(tmp_input)
        depth_order = torch.argsort(res_depth, dim=1).squeeze()
        sorted_alpha = res_alpha[0,:,depth_order,:]
        sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
        sorted_color = colors[depth_order,:]
        alphac = res_alpha[0]*colors[None]
        sorted_alphac = alphac[:,depth_order]
        rgb_color = (sorted_T * sorted_alphac).sum(dim=1)
        res_alpha = res_alpha.detach().cpu().numpy()
        empirical_alpha_lb = np.minimum(empirical_alpha_lb, res_alpha)
        empirical_alpha_ub = np.maximum(empirical_alpha_ub, res_alpha)
        rgb_color = rgb_color.reshape(w, h, -1)[:,:,:3]
        rgb_color = rgb_color.detach().cpu().numpy()
        empirical_lb = np.minimum(empirical_lb, rgb_color)
        empirical_ub = np.maximum(empirical_ub, rgb_color)
        valid_bound = np.all(rgb_color>=tile_color_lb) and np.all(rgb_color<=tile_color_ub)
        if not valid_bound:
            print("Bound Violated")
            break

    diff_compemp_ub = (ub_alpha-empirical_alpha_ub).reshape(w,h,-1)
    diff_compemp_lb = (empirical_alpha_lb-lb_alpha).reshape(w,h,-1)

    # tile_color_ub[:,:,1:] = 0

    plt.figure(1)
    plt.imshow(tile_color_lb)
    plt.title("computed lb alpha-crown")
    plt.figure(2)
    plt.imshow(tile_color_ub)
    plt.title("computed ub alpha-crown")
    plt.figure(3)
    plt.imshow(empirical_lb)
    plt.title("empirical lb")
    plt.figure(4)
    plt.imshow(empirical_ub)
    plt.title("empirical ub")
    plt.figure(5)
    plt.imshow(diff_lb)
    plt.title("diff_lb")
    plt.figure(6)
    plt.imshow(diff_ub)
    plt.title("diff_ub")
    plt.figure(7)
    plt.imshow(diff_compemp_lb)
    plt.title('diff_comp_emp_lb')
    plt.figure(8)
    plt.imshow(diff_compemp_ub)
    plt.title('diff_comp_emp_ub')
    plt.show()