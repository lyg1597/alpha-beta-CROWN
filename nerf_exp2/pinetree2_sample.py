from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
# from rasterize_model import RasterizationModelRGB_notile, DepthModel
from simple_model2_alphatest2 import AlphaModel, DepthModel
from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
# from splat_model import SplatModel
from typing import List, Dict
from scipy.spatial.transform import Rotation 
from collections import defaultdict
import itertools
from img_helper import get_rect 

dt = {
    "transform": [
        [
            0.9990379810333252,
            -0.030679119750857353,
            -0.031336162239313126,
            -79.60553741455078
        ],
        [
            -0.030679119750857353,
            0.021658122539520264,
            -0.999294638633728,
            1.5170279741287231
        ],
        [
            0.031336162239313126,
            0.999294638633728,
            0.020696043968200684,
            -28.584909439086914
        ]
    ],
    "scale": 0.007225637783593824
}

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

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

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


def bounds_union(b1, b2):
    '''
    b1: 2*m
    b2: 2*m
    '''
    b_out = torch.zeros(b1.shape)
    b_out[0,:] = torch.min(torch.stack((b1[0,:],b2[0,:]), dim=0), dim=0).values
    b_out[1,:] = torch.max(torch.stack((b1[1,:],b2[1,:]), dim=0), dim=0).values
    return b_out 

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

def write_value(res: torch.Tensor, fn: str, gt: torch.Tensor = None):
    res_array = res.detach().cpu().numpy()
    if gt is not None:
        gt_array = gt.detach().cpu().numpy()
    with open(fn, 'w+') as f:
        if gt is not None:
            f.write("lb, ub, gt\n")
        else:
            f.write("lb, ub\n")
        for i in range(res.shape[1]):
            val0 = res_array[0,i]
            val1 = res_array[1,i]
            if gt is not None:
                f.write(f"{val0:1.4f}, {val1:1.4f}, {gt[i]:1.4f}\n")
            else:
                f.write(f"{val0:1.4f}, {val1:1.4f}\n")

if __name__ == "__main__":
    transform = np.array(dt['transform'])
    transform_ap = np.vstack((transform, np.array([0,0,0,1])))
    scale = dt['scale']

    script_dir = os.path.dirname(os.path.realpath(__file__))
    # output_folder = os.path.join(script_dir, '/home/younger/work/nerfstudio/outputs/triangular_data4/splatfacto/2025-01-18_190447')
    # checkpoint = "step-000059999.ckpt"
    output_folder = os.path.join(script_dir, '/home/younger/work/nerfstudio/outputs/triangular_data4/splatfacto/2025-01-19_232156')
    checkpoint = "step-000029999.ckpt"
    
    checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    res = torch.load(checkpoint_fn)
    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']
    colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

    scale_mask = torch.all(scales>-7.5, dim=1)

    # Filter unnecessary gaussians 
    # color_mask = torch.norm(colors, dim=1)>0.1
    # means_trans = torch.inverse(torch.tensor(transform_ap).to(means.device))@means.transpose(0,1)/scale

    means = means[scale_mask,:]
    quats = quats[scale_mask,:]
    opacities = opacities[scale_mask,:]
    scales = scales[scale_mask,:]
    colors = colors[scale_mask,:]


    camera_pose = np.array([
        [
            0.0,
            0.0,
            1.0,
            200.0
        ],
        [
            0.0,
            1.0,
            0.0,
           5.0
        ],
        [
            -1.0,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])

    # transform = np.array(dt['transform'])
    # scale = dt['scale']
    camera_pose_transformed = transform@camera_pose
    camera_pose_transformed[:3,3] *= scale 
    camera_pose_transformed = torch.Tensor(camera_pose_transformed)[None].to(means.device)

    width=48
    height=48
    f = 80

    # eps = torch.Tensor([[0,0,0,0.0001,0.0001,0.0001]]).to(means.device)
    eps = 0.00001
    tile_size = 16
    gauss_step = 5000000

    # camera_to_worlds = torch.Tensor(camera_pose)[None].to(means.device)
    camera_to_world = [
        [0.09323342123480016,
        -0.21989920695796833,
        0.9710570878913922,
        0.3464003611615004,],
        [0.9956442784272173,
        0.02059164687201054,
        -0.09093104483202696,
        -0.0148054846552511,],
        [-2.7755575615628914e-16,
        0.975305245991405,
        0.22086121692512006,
        -0.13118252322656915,],
        [0.0,
        0.0,
        0.0,
        1.0]
    ]
    camera_to_world = torch.Tensor(camera_pose_transformed)[None].to(means.device)

    view_mats = get_viewmat(camera_pose_transformed)
    Ks = torch.tensor([[
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]
    ]]).to(torch.device('cuda'))

    camera_pos = view_mats[0,:3,3].detach().cpu().numpy()
    camera_ori = Rotation.from_matrix(view_mats[0,:3,:3].detach().cpu().numpy()).as_euler('xyz')
    cam_inp = [
        camera_ori[0], 
        camera_ori[1], 
        camera_ori[2], 
        camera_pos[0], 
        camera_pos[1], 
        camera_pos[2]
    ]
    cam_inp = torch.Tensor(cam_inp)[None].to('cuda')


    with torch.no_grad():
        res = rasterize_gaussians_pytorch_rgb(
            means, 
            quats/ quats.norm(dim=-1, keepdim=True),
            torch.exp(scales),
            torch.sigmoid(opacities).squeeze(-1),
            colors,
            view_mats, 
            Ks,
            width,
            height
        )
    res_rgb = res['render']
    print(res_rgb.shape)
    res_rgb = res_rgb[:,...,:3]
    res_rgb = res_rgb.detach().cpu().numpy()
    plt.figure(0)
    plt.imshow(res_rgb)
    plt.show()

    # Get all the pix_coord 
    empirical_lb = np.zeros((48,48,3))+1e10
    empirical_ub = np.zeros((48,48,3))-1e10
    for i in range(100):
        pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to(means.device)
        # Get the rectangles of gaussians under uncertainty 
        rect = get_rect(
            cam_inp,
            means,
            scales,
            quats,
            f,
            f,
            width,
            height 
        )
        render_color = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
        emp_pos = torch.rand((1,3))*eps*2-eps
        cam_inp[:,3:] = cam_inp[:,3:]+emp_pos.to('cuda')

        for h in range(0, height, tile_size):
            for w in range(0, width, tile_size):
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+tile_size-1), rect[1][..., 1].clip(max=h+tile_size-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
                if not in_mask.sum() > 0:
                    continue
                means_strip = means[in_mask]
                quats_strip = quats[in_mask]
                opacities_strip = opacities[in_mask]
                scales_strip = scales[in_mask]
                colors_strip = colors[in_mask]
                # means_strip = means
                # quats_strip = quats
                # opacities_strip = opacities
                # scales_strip = scales
                # colors_strip = colors


                tile_coord = pix_coord[h:h+tile_size, w:w+tile_size].flatten(0,-2)
                N = means_strip.shape[0]

                overall_alpha = torch.zeros((1,tile_size*tile_size, 0, 1)).to(means.device)
                overall_depth = torch.zeros((1,0)).to(means.device)

                for j in range(0, N, gauss_step):
                    data_pack = {
                        'opacities': torch.Tensor(opacities_strip[j:j+gauss_step]),
                        'means': torch.Tensor(means_strip[j:j+gauss_step]),
                        'scales':torch.Tensor(scales_strip[j:j+gauss_step]),
                        'quats':torch.Tensor(quats_strip[j:j+gauss_step]),
                        # 'tile_coords':torch.Tensor(tile_coords)
                    } 

                    model_alpha = AlphaModel(
                        data_pack = data_pack,
                        fx = f,
                        fy = f,
                        width = width,
                        height = height,
                        tile_coord = tile_coord 
                    )

                    model_depth = DepthModel(model_alpha)

                    alpha_res = model_alpha(cam_inp)
                    depth_res = model_depth(cam_inp)

                    overall_alpha = torch.cat((overall_alpha, alpha_res), dim=2)
                    overall_depth = torch.cat((overall_depth, depth_res), dim=1)
                depth_order = torch.argsort(overall_depth, dim=1).squeeze()
                sorted_alpha = overall_alpha[0,:,depth_order,:]
                sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
                sorted_color = colors_strip[depth_order,:]
                rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
                rgb_color = rgb_color.reshape(tile_size, tile_size, -1)[:,:,:3]
                rgb_color = rgb_color.detach().cpu().numpy()
                render_color[h:h+tile_size, w:w+tile_size] = rgb_color
        empirical_lb = np.minimum(empirical_lb, render_color)
        empirical_ub = np.maximum(empirical_ub, render_color)
        plt.figure(1)
        plt.imshow(render_color)
        plt.show()
    # plt.figure(1)
    # plt.imshow(empirical_lb)
    # plt.savefig('res_lb_emp.png')
    # plt.figure(2)
    # plt.imshow(empirical_ub)
    # plt.savefig('res_ub_emp.png')
    # plt.show()
                # inp_alpha = torch.clone(cam_inp)
                # print(">>>>>> Starting Bounded Module")
                # model_alpha_bounded = BoundedModule(model_alpha, inp_alpha, device=means.device)
                # print(">>>>>> Starting PerturbationLpNorm")
                # ptb_alpha = PerturbationLpNorm(norm=np.inf, x_L=inp_alpha-eps, x_U=inp_alpha+eps)
                # print(">>>>>> Starting BoundedTensor")
                # inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
                # prediction = model_alpha_bounded(inp_alpha)
                # lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(inp_alpha, ), method='ibp')
                # bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)

                # inp_depth = torch.clone(cam_inp)
                # print(">>>>>> Starting Bounded Module")
                # model_depth_bounded = BoundedModule(model_depth, inp_depth, device=colors.device)
                # print(">>>>>> Starting PerturbationLpNorm")
                # ptb_depth = PerturbationLpNorm(norm=np.inf, x_L=inp_depth-eps, x_U=inp_depth+eps)
                # print(">>>>>> Starting BoundedTensor")
                # inp_depth = BoundedTensor(inp_depth, ptb_depth)
                # prediction = model_depth_bounded(inp_depth)
                # required_A = defaultdict(set)
                # required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
                # lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, ), method='crown', return_A=True, needed_A_dict=required_A)
 
            # concrete_before, possible_before = get_elem_before_linear(ptb_depth, A_depth, model_depth_bounded)
            # print(concrete_before, possible_before)
            # res_T = computeT(concrete_before, possible_before, bounds_alpha)
            
            # res_2d = colors
            # bounds_res_2d = torch.stack((res_2d, res_2d), dim=0)
            # bounds_res_2d = bounds_res_2d[:,None]
            # tile_color = (res_T*bounds_alpha*bounds_res_2d).sum(dim=2)

    # overall_mask = res['overall_mask']
    # means = means[overall_mask].detach()
    # colors = colors[overall_mask].detach()
    # quats = quats[overall_mask].detach()
    # opacities = opacities[overall_mask].detach()
    # scales = scales[overall_mask].detach()

    # means = means[:5000]
    # colors = colors[:5000]
    # quats = quats[:5000]
    # opacities = opacities[:5000]
    # scales = scales[:5000]

    # data_pack = {
    #     'opacities': torch.Tensor(opacities),
    #     'means': torch.Tensor(means),
    #     'scales':torch.Tensor(scales),
    #     'quats':torch.Tensor(quats),
    #     # 'tile_coords':torch.Tensor(tile_coords)
    # }

    # model_alpha = AlphaModel(
    #     data_pack = data_pack,
    #     fx = f,
    #     fy = f,
    #     width = width,
    #     height = height
    # )

    # transform = np.array(dt['transform'])
    # scale = dt['scale']
    # camera_pose_transformed = transform@camera_pose
    # camera_pose_transformed[:3,3] *= scale 
    # camera_pose_transformed = torch.Tensor(camera_pose_transformed)
    # camera_pose_transformed = get_viewmat(camera_pose_transformed[None])
    
    # camera_pos = camera_pose_transformed[0,:3,3].detach().cpu().numpy()
    # camera_ori = Rotation.from_matrix(camera_pose_transformed[0,:3,:3]).as_euler('xyz')
    # camera_pose_transformed = [
    #     camera_ori[0], 
    #     camera_ori[1], 
    #     camera_ori[2], 
    #     camera_pos[0], 
    #     camera_pos[1], 
    #     camera_pos[2]
    # ]
    # camera_pose_transformed = torch.Tensor(camera_pose_transformed)[None].to('cuda')

    # model_depth = DepthModel(model_alpha)
    
    # res_alpha = model_alpha(camera_pose_transformed)
    # print("###### Alpha")
    # res_depth = model_depth(camera_pose_transformed)
    # print("###### Depth")
    # depth_order = torch.argsort(res_depth, dim=1).squeeze()
    # sorted_alpha = res_alpha[0,:,depth_order,:]
    # sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    # sorted_color = colors[depth_order,:]
    # rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
    # rgb_color = rgb_color.reshape(width, height, -1)[:,:,:3]
    # rgb_color = rgb_color.detach().cpu().numpy()
    # plt.figure(0)
    # plt.imshow(rgb_color)
    # plt.show()

    # # torch.onnx.export(model_alpha, view_mats, 'model.onnx') 

    # # my_input = torch.clone(res_2d[model.overall_mask])
    # inp_alpha = torch.clone(camera_pose_transformed)
    # print(">>>>>> Starting Bounded Module")
    # model_alpha_bounded = BoundedModule(model_alpha, inp_alpha, device=means.device)
    # print(">>>>>> Starting PerturbationLpNorm")
    # ptb_alpha = PerturbationLpNorm(norm=np.inf, eps=0.00001)
    # print(">>>>>> Starting BoundedTensor")
    # inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
    # prediction = model_alpha_bounded(inp_alpha)
    # lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(inp_alpha, ), method='ibp')
    # bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)
    # # bounds_alpha = 
    # # lb2, ub2 = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    # # print(len(torch.where((lb<lb2)|(ub>ub2))[0]))
    # # print(torch.max(lb2-lb), torch.max(ub-ub2))
    
    # print(">>>>>> Done")
    # print(lb_alpha.shape)
    # print(ub_alpha.shape)

    # # model_depth = DepthModel(model_alpha)
    # # view_mats = model_alpha.viewmat
    # # with torch.no_grad():
    # #     res = model_depth(view_mats)
    # # print(res.shape)
    # inp_depth = torch.clone(camera_pose_transformed)
    # print(">>>>>> Starting Bounded Module")
    # model_depth_bounded = BoundedModule(model_depth, inp_depth, device=colors.device)
    # print(">>>>>> Starting PerturbationLpNorm")
    # ptb_depth = PerturbationLpNorm(norm=np.inf, eps=0.00001)
    # print(">>>>>> Starting BoundedTensor")
    # inp_depth = BoundedTensor(inp_depth, ptb_depth)
    # prediction = model_depth_bounded(inp_depth)
    # required_A = defaultdict(set)
    # required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
    # lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, ), method='crown', return_A=True, needed_A_dict=required_A)
    
    # lb_depth = lb_depth.detach().cpu().numpy()    
    # ub_depth = ub_depth.detach().cpu().numpy()    
    # bounds_depth = np.vstack((lb_depth, ub_depth)).T
    # bounds_depth = bounds_depth.tolist()
    # bounds_depth = [elem+[i] for i, elem in enumerate(bounds_depth)]
    # # sorted_bounds = sort_bounds(bounds_depth)

    # # concrete_before, possible_before = get_elem_before(bounds_depth)
    # concrete_before, possible_before = get_elem_before_linear(ptb_depth, A_depth, model_depth_bounded)
    # print(concrete_before, possible_before)
    # res_T = computeT(concrete_before, possible_before, bounds_alpha)
    
    # res_2d = colors
    # bounds_res_2d = torch.stack((res_2d, res_2d), dim=0)
    # bounds_res_2d = bounds_res_2d[:,None]
    # tile_color = (res_T*bounds_alpha*bounds_res_2d).sum(dim=2)

    # tile_color_lb = tile_color[0,:,:3].reshape((width,height,-1))
    # tile_color_lb = tile_color_lb.detach().cpu().numpy()
    # tile_color_ub = tile_color[1,:,:3].reshape((width,height,-1))
    # tile_color_ub = tile_color_ub.detach().cpu().numpy()

    # empirical_lb = np.zeros(tile_color_lb.shape)+1e10
    # empirical_ub = np.zeros(tile_color_lb.shape)
    # empirical_alpha_lb = np.zeros(lb_alpha.shape)+1e10
    # empirical_alpha_ub = np.zeros(ub_alpha.shape)
    # lb_alpha = lb_alpha.detach().cpu().numpy()
    # ub_alpha = ub_alpha.detach().cpu().numpy()
    # for i in range(1000):
    #     tmp_input = my_input.repeat(1,1)
    #     delta = torch.zeros((1,6))
    #     # delta[:,:3,3] = torch.rand((1,3))*eps*2-eps
    #     delta[:,:3] = torch.rand((1,3))*0.0*2-0.0
    #     delta[:,3:] = torch.rand((1,3))*1.0*2-1.0
    #     delta = delta.to(model_depth.device)
    #     tmp_input = tmp_input+delta 
    #     res_alpha = model_alpha(tmp_input)
    #     res_depth = model_depth(tmp_input)
    #     depth_order = torch.argsort(res_depth, dim=1).squeeze()
    #     sorted_alpha = res_alpha[0,:,depth_order,:]
    #     sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    #     sorted_color = colors[depth_order,:]
    #     alphac = res_alpha[0]*colors[None]
    #     sorted_alphac = alphac[:,depth_order]
    #     rgb_color = (sorted_T * sorted_alphac).sum(dim=1)
    #     res_alpha = res_alpha.detach().cpu().numpy()
    #     empirical_alpha_lb = np.minimum(empirical_alpha_lb, res_alpha)
    #     empirical_alpha_ub = np.maximum(empirical_alpha_ub, res_alpha)
    #     rgb_color = rgb_color.reshape(width, height, -1)[:,:,:3]
    #     rgb_color = rgb_color.detach().cpu().numpy()
    #     empirical_lb = np.minimum(empirical_lb, rgb_color)
    #     empirical_ub = np.maximum(empirical_ub, rgb_color)
    #     valid_bound = np.all(rgb_color>=tile_color_lb) and np.all(rgb_color<=tile_color_ub)
    #     if not valid_bound:
    #         print("Bound Violated")
    #         break

    # diff_compemp_ub = (ub_alpha-empirical_alpha_ub).reshape(width,height,-1)
    # diff_compemp_lb = (empirical_alpha_lb-lb_alpha).reshape(width,height,-1)

    # tile_color_ub[:,:,1:] = 0

    # plt.figure(1)
    # plt.imshow(tile_color_lb)
    # plt.title("computed lb alpha-crown")
    # plt.figure(2)
    # plt.imshow(tile_color_ub)
    # plt.title("computed ub alpha-crown handle 0")
    # plt.figure(3)
    # plt.imshow(empirical_lb)
    # plt.title("empirical lb")
    # plt.figure(4)
    # plt.imshow(empirical_ub)
    # plt.title("empirical ub")
    # plt.figure(5)
    # plt.imshow(diff_compemp_lb)
    # plt.title('diff_comp_emp_lb')
    # plt.figure(6)
    # plt.imshow(diff_compemp_ub)
    # plt.title('diff_comp_emp_ub')
    # plt.show()