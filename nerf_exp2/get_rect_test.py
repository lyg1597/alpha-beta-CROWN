from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
import numpy as np 
from simple_model2_alphatest4 import AlphaModel, DepthModel, MeanModel
from scipy.spatial.transform import Rotation 

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

def get_rect_set(
    # Input perturbation 
    x_L,
    x_U,
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
    means_strip = means#[:100]
    scales_strip = scales#[:100]
    quats_strip = quats#[:100]

    lb_mean = torch.zeros((1,0,3)).to(means_strip.device)
    ub_mean = torch.zeros((1,0,3)).to(means_strip.device)
    N = means_strip.shape[0]

    inp_mean = torch.clone((x_L+x_U)/2).repeat((N,1))
    model_mean = MeanModel(means_strip, scales_strip, quats_strip, fx, fy, width, height)
    means_hom_tmp = model_mean.means_hom_tmp.transpose(0,2)
    model_mean_bounded = BoundedModule(model_mean, (inp_mean, means_hom_tmp), device = means.device)
    ptb_mean = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
    inp_mean = BoundedTensor(inp_mean, ptb_mean)
    lb_mean_part, ub_mean_part = model_mean_bounded.compute_bounds(x=(inp_mean, means_hom_tmp), method='crown')
    lb_mean = torch.cat((lb_mean, lb_mean_part.transpose(0,1)), dim=1)
    ub_mean = torch.cat((ub_mean, ub_mean_part.transpose(0,1)), dim=1)

    lb_mean = lb_mean.squeeze()
    ub_mean = ub_mean.squeeze()
    mask = ub_mean[:,2]>=1e-8
    lb_mean = lb_mean[mask]
    ub_mean = ub_mean[mask]
    lb_mean = lb_mean.clip(min=1e-8)
    ub_mean = ub_mean.clip(min=1e-8)
    lb_mean2D = lb_mean[:,:2]/ub_mean[:,2:]
    ub_mean2D = ub_mean[:,:2]/lb_mean[:,2:]

    radii = model_mean.get_radii((x_L+x_U)/2)
    radii = radii.squeeze()[mask]
    rect_min = lb_mean2D-radii[:,None]
    rect_max = ub_mean2D+radii[:,None]
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return (rect_min, rect_max), mask 

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

if __name__ == "__main__":
    transform = np.array(dt['transform'])
    transform_ap = np.vstack((transform, np.array([0,0,0,1])))
    scale = dt['scale']

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/triangular_data4/splatfacto/2025-01-19_232156')
    checkpoint = "step-000029999.ckpt"
    
    checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    res = torch.load(checkpoint_fn)
    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']
    colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

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

    eps = torch.Tensor([[0,0,0,0.0001,0.0001,0.0001]]).to(means.device)
    tile_size_global = 4
    gauss_step = 150
    threshold = tile_size_global**2*gauss_step
    initial_tilesize = 128

    # camera_to_worlds = torch.Tensor(camera_pose)[None].to(means.device)
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

    # Get all the pix_coord 
    pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to(means.device)
    # Get the rectangles of gaussians under uncertainty 
    rect, mask = get_rect_set(
        cam_inp-eps,
        cam_inp+eps,
        means,
        scales,
        quats,
        f,
        f,
        width,
        height 
    )
