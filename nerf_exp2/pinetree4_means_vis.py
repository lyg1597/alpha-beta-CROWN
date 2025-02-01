from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
# from rasterize_model import RasterizationModelRGB_notile, DepthModel
from simple_model2_alphatest4_2 import AlphaModel, DepthModel, MeanModel
from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
# from splat_model import SplatModel
from typing import List, Dict
from scipy.spatial.transform import Rotation 
from collections import defaultdict
import itertools
# from img_helper import get_rect 

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

def get_rect(
    # Input perturbation 
    cam_inp,
    # Info of gaussians 
    means,
    scales,
    quats,
    # Image width and height
    fx,
    fy,
    width,
    height,
):
    model_mean = MeanModel(means, scales, quats, fx, fy, width, height)
    means_hom_tmp = model_mean.means_hom_tmp.transpose(0,2)
    means_hom_tmp[:,1,:] += 0.02
    means_proj_hom = model_mean(means_hom_tmp, cam_inp)
    mask = (means_proj_hom[:,0,2]>0.01) & (means_proj_hom[:,0,2]<10000000000)
    means2D = (means_proj_hom[mask,:,:2]/means_proj_hom[mask,:,2:]).squeeze()

    radii = model_mean.get_radii(means_hom_tmp, cam_inp)
    radii = radii.squeeze()[mask]
    rect_min = means2D-radii[:,None] 
    rect_max = means2D+radii[:,None]
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return (rect_min, rect_max), mask


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

    # Tree1
    mask_tree1 = (means[:,0]>=0.476) & (means[:,0]<=0.508) & (means[:,1]>=-0.1) & (means[:,1]<=-0.06) & (means[:,2]>=-0.162) & (means[:,2]<=-0.06)
    # means = means[~mask]
    # quats = quats[~mask]
    # opacities = opacities[~mask]
    # scales = scales[~mask]
    # colors = colors[~mask]

    # Tree2
    mask_tree2 = (means[:,0]>=0.6) & (means[:,0]<=0.64) & (means[:,1]>=-0.1) & (means[:,1]<=-0.06) & (means[:,2]>=-0.158) & (means[:,2]<=-0.055)
    # means = means[~mask]
    # quats = quats[~mask]
    # opacities = opacities[~mask]
    # scales = scales[~mask]
    # colors = colors[~mask]
    mask_trees = mask_tree1 | mask_tree2
    means = means[mask_trees]
    quats = quats[mask_trees]
    opacities = opacities[mask_trees]
    scales = scales[mask_trees]
    colors = colors[mask_trees]

    # mask = (means[:,0]>=-0.02) & (means[:,0]<=0.85) & (means[:,1]>=-0.151) & (means[:,1]<=-0.098) & (means[:,2]>=-0.17) & (means[:,2]<=-0.071)
    # means = means[mask]
    # quats = quats[mask]
    # opacities = opacities[mask]
    # scales = scales[mask]
    # colors = colors[mask]
    # opacities[mask] = -10

    # Filter unnecessary gaussians 
    # color_mask = torch.norm(colors, dim=1)>0.1
    # means_trans = torch.inverse(torch.tensor(transform_ap).to(means.device))@means.transpose(0,1)/scale

    # means = means[color_mask,:]
    # quats = quats[color_mask,:]
    # opacities = opacities[color_mask,:]
    # scales = scales[color_mask,:]
    # colors = colors[color_mask,:]


    camera_pose = np.array([
        [
            0.0,
            0.0,
            1.0,
            194.5
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

    width=96
    height=96
    f = 120

    eps = 0.001
    tile_size = 4
    gauss_step = 10

    # camera_to_worlds = torch.Tensor(camera_pose)[None].to(means.device)
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

    cam_inp = view_mats

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
            height,
            eps2d=0.0,
        )
    res_rgb = res['render']
    print(res_rgb.shape)
    res_rgb = res_rgb[:,...,:3]
    res_rgb = res_rgb.detach().cpu().numpy()
    plt.figure(0)
    plt.imshow(res_rgb)
    plt.show()

    # Get all the pix_coord 
    pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to(means.device)
    # Get the rectangles of gaussians under uncertainty 
    rect, mask = get_rect(
        cam_inp,
        means,
        scales,
        quats,
        f,
        f,
        width,
        height 
    )

    means = means[mask]
    quats = quats[mask]
    opacities = opacities[mask]
    scales = scales[mask]
    colors = colors[mask]
    # mask_trees = mask_trees[mask]

    render_color = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
    for h in range(0, height, 8):
        for w in range(0, width, 8):
            # if h!=32 and w!=24:
            #     continue
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+8-1), rect[1][..., 1].clip(max=h+8-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
            if not in_mask.sum() > 0:
                continue
            means_strip = means[in_mask]
            quats_strip = quats[in_mask]
            opacities_strip = opacities[in_mask]
            scales_strip = scales[in_mask]
            colors_strip = colors[in_mask]
            # mask_trees_strip = mask_trees[in_mask]
            # means_strip = means
            # quats_strip = quats
            # opacities_strip = opacities
            # scales_strip = scales
            # colors_strip = colors


            tile_coord = pix_coord[h:h+8, w:w+8].flatten(0,-2)
            N = means_strip.shape[0]

            overall_alpha = torch.zeros((1,8*8, 0, 1)).to(means.device)
            overall_depth = torch.zeros((1,0)).to(means.device)

            for j in range(0, N, 10000000000):
                data_pack = {
                    'opacities': torch.Tensor(opacities_strip[j:j+10000000000]),
                    'means': torch.Tensor(means_strip[j:j+10000000000]),
                    'scales':torch.Tensor(scales_strip[j:j+10000000000]),
                    'quats':torch.Tensor(quats_strip[j:j+10000000000]),
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

                means_hom_tmp = model_alpha.means_hom_tmp.transpose(0,2)
                means_hom_tmp[:,1,:] += 0.02
                # if torch.any(mask_trees_strip):
                #     print("aa") 
                # means_hom_tmp[mask_trees_strip,1,:] = means_hom_tmp[mask_trees_strip,1,:]+0.1
                # means_hom_tmp = means_hom_tmp + 0.03
                cov_world = model_alpha.cov_world 
                opacities_rast = model_alpha.opacities_rast.transpose(0,2)[:,:,0,0]

                alpha_res = model_alpha(means_hom_tmp, cam_inp, cov_world, opacities_rast)
                alpha_res = alpha_res*(-0.5)
                alpha_res = torch.exp(alpha_res)
                alpha_res = alpha_res*opacities_rast
                depth_res = model_depth(means_hom_tmp, cam_inp)

                alpha_res = alpha_res.transpose(0,1)[None,:,:,None]
                depth_res = depth_res.transpose(0,1)

                overall_alpha = torch.cat((overall_alpha, alpha_res), dim=2)
                overall_depth = torch.cat((overall_depth, depth_res), dim=1)
            # mask = (overall_depth[0,:]>0.01)&(overall_depth[0,:]<10000000000)
            # overall_depth = overall_depth[:,mask]
            # overall_alpha = overall_alpha[:,:,mask]
            # colors_strip = colors_strip[mask]
            depth_order = torch.argsort(overall_depth, dim=1).squeeze()
            sorted_alpha = overall_alpha[0,:,depth_order,:]
            sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
            sorted_color = colors_strip[depth_order,:]
            rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
            rgb_color = rgb_color.reshape(8, 8, -1)[:,:,:3]
            rgb_color = rgb_color.detach().cpu().numpy()
            render_color[h:h+8, w:w+8] = rgb_color

    plt.imshow(render_color)
    plt.show()
                