from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
# from pathlib import Path 
# import yaml 
# from splatfactoenv.splatfactoenv_model import SplatfactoEnvModel
# from nerfstudio.data.scene_box import SceneBox
import numpy as np 
# from nerfstudio.cameras.cameras import Cameras, CameraType
import matplotlib.pyplot as plt 
from rasterize_model import RasterizationModel, RasterizationModel_notile, RasterizationModelRGB_notile, DepthModel
from gsplat import rasterization
# import cv2 
from rasterization_pytorch import rasterize_gaussians_pytorch, rasterize_gaussians_pytorch_rgb
from splat_model import SplatModel
from typing import List

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
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/gazebo5_transformed_env-1/splatfacto-env-rgb/2024-11-18_154538/')
    checkpoint = "step-000029999.ckpt"
    
    camera_pose = np.array([
                [
                    -0.23762398510466104,
                    0.44276476982071006,
                    -0.864577469234882,
                    -2230.7194253135594
                ],
                [
                    -2.9884813341042206e-16,
                    0.8900715974578106,
                    0.45582074480973456,
                    358.8874872340502
                ],
                [
                    0.9713572163231092,
                    0.10831394187506413,
                    -0.21150236001639652,
                    -166.52500219585227
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
    ])

    fn = "frames_00775_gs.png"

    width=16
    height=16
    f = 1200.0

    model = SplatModel(
        output_folder=output_folder,
        camera_pose = camera_pose,
        checkpoint=checkpoint,
        width=width,
        height=height,
        fx = f,
        fy = f,
        use_sh = False
    )
    my_input = torch.Tensor(np.array([[0.0,0.0]])).to(torch.device('cuda'))
    with torch.no_grad():
        res_2d = model(my_input)
    print(res_2d.shape)

    # print(">>>>>> Starting Bounded Module")
    # model_bounded = BoundedModule(model, my_input, device=res_2d.device)
    # print(">>>>>> Starting PerturbationLpNorm")
    # ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # print(">>>>>> Starting BoundedTensor")
    # my_input = BoundedTensor(my_input, ptb)
    # prediction = model_bounded(my_input)
    # lb, ub = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    # print(">>>>>> Done")
    # print(lb.shape)
    # print(ub.shape)


    # view_mats = torch.cat((model.camera.camera_to_worlds, addon), dim=1)
    view_mats = get_viewmat(model.camera.camera_to_worlds)
    Ks = torch.tensor([[
        [model.camera.fx, 0, model.camera.cx],
        [0, model.camera.fy, model.camera.cy],
        [0, 0, 1]
    ]]).to(torch.device('cuda'))

    # with torch.no_grad():
    #     res = rasterization(
    #         model.means,
    #         model.quats/ model.quats.norm(dim=-1, keepdim=True),
    #         torch.exp(model.scales),
    #         torch.sigmoid(model.opacities).squeeze(-1),
    #         res_2d,
    #         view_mats,      # 
    #         Ks,
    #         model.width,
    #         model.height,
    #         sh_degree=None,
    #     )

    # rendered = res[0]
    # print(rendered.shape)
    # rendered = rendered.detach().cpu().numpy()[0]
    # print(rendered.shape)
    # rendered = rendered[:,...,:3]
    # image_uint8 = (np.clip(rendered, 0, 1)*255).astype(np.uint8)
    # # image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    # # cv2.imwrite(fn, image_bgr)
    # plt.imshow(rendered)
    # plt.show()

    with torch.no_grad():
        res = rasterize_gaussians_pytorch_rgb(
            model.means, 
            model.quats/ model.quats.norm(dim=-1, keepdim=True),
            torch.exp(model.scales),
            torch.sigmoid(model.opacities).squeeze(-1),
            res_2d,
            view_mats, 
            Ks,
            model.width,
            model.height
        )
    res_rgb = res['render']
    print(res_rgb.shape)
    res_rgb = res_rgb[:,...,:3]
    res_rgb = res_rgb.detach().cpu().numpy()
    plt.figure(0)
    plt.imshow(res_rgb)

    # # with torch.no_grad():
    # #     res, colors_gt, rasterizer_input = model.model.get_outputs(
    # #         model.camera,
    # #         debug = True
    # #     )
    # # rendered = res['rgb']
    # # print(rendered.shape)
    # # rendered = rendered.detach().cpu().numpy()
    # # plt.imshow(rendered)
    # # plt.show()

    # # model_bounded = BoundedModule(model, my_input)
    # # ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # # my_input = BoundedTensor(my_input, ptb)
    # # prediction = model_bounded(my_input)
    # # lb, ub = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    # # print(lb,ub)

    model_alpha = RasterizationModelRGB_notile(
        output_folder=output_folder,
        camera_pose = camera_pose,
        checkpoint = checkpoint,
        width=width,
        height=height,
        fx = f,
        fy = f,
        tile_size=8
    )
    overall_mask = res['overall_mask']
    model_alpha.strip_gaussians(
        eps = torch.FloatTensor([10, 10, 10, 0.001, 0.001, 0.001]),
        near_plane = 0.001, overall_mask = overall_mask
    )
    print(model_alpha.means.shape)
    # model.to(torch.device('cuda'))
    # # my_input = torch.rand((602465, 16, 3)).to(torch.device('cuda'))
    # # my_input[:,0,:] = 0
    # tmp = res_2d[model.overall_mask]
    # print(tmp.shape)
    # with torch.no_grad():
    #     res = model(tmp)
    # print(res.shape)
    # # res = res[:,...,:3]
    # res = res.detach().cpu().numpy()
    # res = res.reshape((60, 80, -1))
    # res = res[:,...,:3]
    # plt.imshow(res)
    # plt.show()
    view_mats = model_alpha.viewmat
    with torch.no_grad():
        res_alpha = model_alpha(view_mats)
    print(res_alpha.shape)


    model_depth = DepthModel(model_alpha)
    view_mats = model_alpha.viewmat
    with torch.no_grad():
        res_depth = model_depth(view_mats)
    print(res_depth.shape)
    depth_order = torch.argsort(res_depth, dim=1).squeeze()
    sorted_alpha = res_alpha[0,:,depth_order,:]
    sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    used_color = res_2d[overall_mask,:]
    sorted_color = used_color[depth_order,:]
    rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
    rgb_color = rgb_color.reshape(16, 16, -1)[:,:,:3]
    rgb_color = rgb_color.detach().cpu().numpy()
    plt.figure(3)
    plt.imshow(rgb_color)
    plt.show()
    # torch.onnx.export(model_alpha, view_mats, 'model.onnx') 

    # my_input = torch.clone(res_2d[model.overall_mask])
    my_input = torch.clone(view_mats)
    print(">>>>>> Starting Bounded Module")
    model_alpha_bounded = BoundedModule(model_alpha, my_input, device=res_2d.device)
    print(">>>>>> Starting PerturbationLpNorm")
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.00001)
    print(">>>>>> Starting BoundedTensor")
    my_input = BoundedTensor(my_input, ptb)
    prediction = model_alpha_bounded(my_input)
    lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(x=(my_input, ), method='ibp')
    bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)
    # bounds_alpha = 
    # lb2, ub2 = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    # print(len(torch.where((lb<lb2)|(ub>ub2))[0]))
    # print(torch.max(lb2-lb), torch.max(ub-ub2))
    
    print(">>>>>> Done")
    print(lb_alpha.shape)
    print(ub_alpha.shape)

    model_depth = DepthModel(model_alpha)
    view_mats = model_alpha.viewmat
    with torch.no_grad():
        res = model_depth(view_mats)
    print(res.shape)
    my_input = torch.clone(view_mats)
    print(">>>>>> Starting Bounded Module")
    model_depth_bounded = BoundedModule(model_depth, my_input, device=res_2d.device)
    print(">>>>>> Starting PerturbationLpNorm")
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.00001)
    print(">>>>>> Starting BoundedTensor")
    my_input = BoundedTensor(my_input, ptb)
    prediction = model_depth_bounded(my_input)
    lb_depth, ub_depth = model_depth_bounded.compute_bounds(x=(my_input, ), method='ibp')
    
    lb_depth = lb_depth.detach().cpu().numpy()    
    ub_depth = ub_depth.detach().cpu().numpy()    
    bounds_depth = np.vstack((lb_depth, ub_depth)).T
    bounds_depth = bounds_depth.tolist()
    bounds_depth = [elem+[i] for i, elem in enumerate(bounds_depth)]
    sorted_bounds = sort_bounds(bounds_depth)

    # print(sorted_bounds)
    set_order = get_set_order(sorted_bounds)
    # Check set order
    depth_order_array = depth_order.detach().cpu().numpy()
    for i in range(len(set_order)):
        val = depth_order_array[i]
        if val in set_order[i]:
            continue
        else:
            print(f"Set order violated: {i}, {val}, {set_order[i]}")
    print(len(set_order))
    print(set_order)
    
    set_sorted_alpha = apply_set_order(set_order, bounds_alpha)
    write_value(set_sorted_alpha[:,0,:,0], 'alpha.txt', sorted_alpha[0,:,0])
    # Check sorted alpha
    set_sorted_T = compute_sortedT(set_sorted_alpha)
    write_value(set_sorted_T[:,0,:,0], 'Talpha.txt', sorted_T[0,:,0])

    res_2d = res_2d[model_alpha.overall_mask,:]
    bounds_res_2d = torch.stack((res_2d, res_2d), dim=0)
    bounds_res_2d = bounds_res_2d[:,None]
    set_sorted_color = apply_set_order(set_order, bounds_res_2d)
    write_value(set_sorted_color[:,0,:,0], 'color.txt')

    tile_color = (set_sorted_T*set_sorted_alpha*set_sorted_color).sum(dim=2)

    tile_color_lb = tile_color[0,:,:3].reshape((16,16,-1))
    tile_color_lb = tile_color_lb.detach().cpu().numpy()
    plt.figure(1)
    plt.imshow(tile_color_lb)
    tile_color_ub = tile_color[1,:,:3].reshape((16,16,-1))
    tile_color_ub = tile_color_ub.detach().cpu().numpy()
    plt.figure(2)
    plt.imshow(tile_color_ub)
    plt.show()

    # res_lb = torch.matmul(model.T_alpha_unsorted, lb).reshape(model.H, model.W, -1)
    # res_ub = torch.matmul(model.T_alpha_unsorted, ub).reshape(model.H, model.W, -1)


    # plt.figure(0)
    # res_lb = res_lb.detach().cpu().numpy()
    # plt.imshow(res_lb)

    # plt.figure(1)
    # res_ub = res_ub.detach().cpu().numpy()
    # plt.imshow(res_ub)

    # plt.show()

    # plt.show()