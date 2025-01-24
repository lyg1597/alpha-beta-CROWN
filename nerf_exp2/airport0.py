from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from simple_model2_alphatest4 import AlphaModel, DepthModel, MeanModel
from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
from scipy.spatial.transform import Rotation 
from collections import defaultdict


method = 'crown'
adaptive_sampling = False 

width=48
height=48
f = 80

eps_lb = torch.Tensor([[0,0,0,-0.0001,-0.0001,-0.0001]]).to('cuda')
eps_ub = torch.Tensor([[0,0,0,0.0001,0.0001,0.0001]]).to('cuda')
tile_size_global = 8
if method == 'ibp' or method=='crown':
    gauss_step = 10000
elif method == 'alpha-crown':
    gauss_step = 1500
threshold = tile_size_global**2*gauss_step
initial_tilesize = 128


def linear_bounds(A,b,x_L, x_U):
    pos_mask = (A>=0).float() 
    neg_mask = 1.0-pos_mask 

    A_pos = A*pos_mask 
    A_neg = A*neg_mask 

    fmin = torch.einsum('iabc,ic->iab',A_pos,x_L)+torch.einsum('iabc,ic->iab',A_neg,x_U)+b 
    fmax = torch.einsum('iabc,ic->iab',A_pos,x_U)+torch.einsum('iabc,ic->iab',A_neg,x_L)+b
    return fmin, fmax

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

def computeT_new_new(
    ptb_depth: PerturbationLpNorm,
    lA_depth: torch.Tensor,
    uA_depth: torch.Tensor,
    lbias_depth: torch.Tensor,
    ubias_depth: torch.Tensor,
    alpha_lb: torch.Tensor,
    alpha_ub: torch.Tensor,
):
    batch_size = 100
    N = lA_depth.shape[1]
    x_L = ptb_depth.x_L[0:1]
    x_U = ptb_depth.x_U[0:1]
    A_lb = alpha_lb.squeeze(0).squeeze(-1)
    A_ub = alpha_ub.squeeze(0).squeeze(-1)
    device = A_lb.device
    P = A_lb.shape[0]
    result_lb = torch.ones((P,N), device = device)                      # P*N
    result_ub = torch.ones((P,N), device = device)                      # P*N
    for i in range(0,N,batch_size):  
        # Get for each gaussian i for all gaussians, what step(d_i-d_j) should be 
        lA_diff = lA_depth[:,:,None]-uA_depth[:,None,i:i+batch_size]             # 1*N*BS*6
        uA_diff = uA_depth[:,:,None]-lA_depth[:,None,i:i+batch_size]             # 1*N*BS*6
        lbias_diff = lbias_depth[:,:,None]-ubias_depth[:,None,i:i+batch_size]    # 1*N*BS
        ubias_diff = ubias_depth[:,:,None]-lbias_depth[:,None,i:i+batch_size]    # 1*N*BS

        diffL_part, _ = linear_bounds(lA_diff, lbias_diff, x_L, x_U)    # 1*N*BS
        _, diffU_part = linear_bounds(uA_diff, ubias_diff, x_L, x_U)    # 1*N*BS

        diffL = torch.minimum(diffL_part, diffU_part)                   # 1*N*BS
        diffU = torch.maximum(diffL_part, diffU_part)                   # 1*N*BS

        # Set diagonal element to 0
        # diffL[:,i,:] = 0
        # diffU[:,i,:] = 0
        actual_bs = min(batch_size, N - i)
        
        # Create indices for diagonal masking [BATCH VERSION]
        batch_indices = torch.arange(i, i + actual_bs, device=device)  # k values: i, i+1, ..., i+bs-1
        offset_indices = torch.arange(0, actual_bs, device=device)     # Positions within current batch
        
        # Vectorized diagonal masking
        diffL[:, batch_indices, offset_indices] = 0  # j=k masking
        diffU[:, batch_indices, offset_indices] = 0

        # assert torch.all(diffL<=diffU)

        step_L = torch.zeros(diffL.shape).to(A_ub.device)               # 1*N*BS
        mask_L = diffL>0
        step_L[mask_L] = 1.0
        step_U = torch.ones(diffU.shape).to(A_ub.device)                # 1*N*BS
        mask_U = diffU<=0
        step_U[mask_U] = 0.0

        term_lb = 1-step_U*A_ub[:,None, i:i+batch_size]                         # P*N
        term_ub = 1-step_L*A_lb[:,None, i:i+batch_size]                         # P*N

        result_lb = result_lb*term_lb.prod(dim=2)
        result_ub = result_ub*term_ub.prod(dim=2)
    return result_lb.unsqueeze(0).unsqueeze(-1), result_ub.unsqueeze(0).unsqueeze(-1)

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

    inp_mean = torch.clone((x_L+x_U)/2)
    model_mean = MeanModel(means_strip, scales_strip, quats_strip, fx, fy, width, height)
    means_hom_tmp = model_mean.means_hom_tmp.transpose(0,2)
    model_mean_bounded = BoundedModule(model_mean, (inp_mean, means_hom_tmp), device = means.device)
    ptb_mean = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
    inp_mean = BoundedTensor(inp_mean, ptb_mean)
    lb_mean, ub_mean = model_mean_bounded.compute_bounds(x=(inp_mean, means_hom_tmp), method='ibp')

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
            0.9999805688858032,
            -0.004387019667774439,
            -0.004431084264069796,
            -0.17920809984207153
        ],
        [
            -0.004387019667774439,
            0.010003387928009033,
            -0.9999403953552246,
            -1.9760571718215942
        ],
        [
            0.004431084264069796,
            0.9999403953552246,
            0.00998389720916748,
            -73.62712860107422
        ]
    ],
    "scale": 0.004533926584786731
}

def compute_tile_color(
    # Input info
    cam_inp,
    eps_lb,
    eps_ub,

    # Gaussian input 
    means_strip,
    opacities_strip,
    scales_strip,
    quats_strip,
    colors_strip,

    # Add camera info 
    f,
    wbl,
    hbl,
    wtr,
    htr,
    width,
    height,

    # Add parameters
    pix_coord,
    tile_size,
    gauss_step,
):
    tile_coord = pix_coord[hbl:htr, wbl:wtr].flatten(0,-2)
    N = means_strip.shape[0]

    overall_alpha_lb = torch.zeros((1,(htr-hbl)*(wtr-wbl), 0, 1)).to(means_strip.device)
    overall_alpha_ub = torch.zeros((1,(htr-hbl)*(wtr-wbl), 0, 1)).to(means_strip.device)

    overall_depth_lA = torch.zeros((1,0,6)).to(means_strip.device)
    overall_depth_uA = torch.zeros((1,0,6)).to(means_strip.device)
    overall_depth_lbias = torch.zeros((1,0)).to(means_strip.device)
    overall_depth_ubias = torch.zeros((1,0)).to(means_strip.device)
    
    for j in range(0, N, gauss_step):
        print(f">>>>>>>> Computation Progress {j}/{N}")
        data_pack = {
            'opacities': torch.Tensor(opacities_strip[j:j+gauss_step]),
            'means': torch.Tensor(means_strip[j:j+gauss_step]),
            'scales':torch.Tensor(scales_strip[j:j+gauss_step]),
            'quats':torch.Tensor(quats_strip[j:j+gauss_step]),
        } 

        model_alpha = AlphaModel(
            data_pack = data_pack,
            fx = f,
            fy = f,
            width = width,
            height = height,
            tile_coord = tile_coord,
        )
        means_hom_tmp = model_alpha.means_hom_tmp.transpose(0,2)
        cov_world = model_alpha.cov_world 
        opacities_rast = model_alpha.opacities_rast.transpose(0,2)[:,:,0,0]
        BS = means_hom_tmp.shape[0]
        # torch.onnx.export(model_alpha, cam_inp, 'pinetree2.model')

        model_depth = DepthModel(model_alpha)

        inp_alpha = torch.clone(cam_inp).repeat(BS, 1)
        # print(">>>>>> Starting Bounded Module")
        model_alpha_bounded = BoundedModule(
            model_alpha, 
            (inp_alpha, means_hom_tmp, cov_world, opacities_rast), 
            device=means_strip.device,
            bound_opts= {
                'conv_mode': 'matrix',
                'optimize_bound_args': {
                    'iteration': 100, 
                    # 'lr_alpha':0.02, 
                    'early_stop_patience':5},
            }, 
        )
        # model_alpha_bounded(inp_alpha)
        # model_alpha_bounded.visualize('pinetree2')
        # print(f"time for create bounded model {time.time()-tmp}")
        # print(">>>>>> Starting PerturbationLpNorm")
        ptb_alpha = PerturbationLpNorm(norm=np.inf, x_L=inp_alpha+eps_lb, x_U=inp_alpha+eps_ub)
        # print(">>>>>> Starting BoundedTensor")
        inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
        # prediction = model_alpha_bounded(inp_alpha)
        # tmp = time.time()
        lb_alpha, ub_alpha = model_alpha_bounded.compute_bounds(
            x=(inp_alpha, means_hom_tmp, cov_world, opacities_rast), 
            method=method
        )
        # print(f'time for compute bound {time.time()-tmp}')
        # bounds_alpha = torch.cat((lb_alpha, ub_alpha), dim=0)
        lb_alpha = lb_alpha.transpose(0,1)[None,:,:,None]
        ub_alpha = ub_alpha.transpose(0,1)[None,:,:,None]
        overall_alpha_lb = torch.cat((overall_alpha_lb, lb_alpha), dim=2)
        overall_alpha_ub = torch.cat((overall_alpha_ub, ub_alpha), dim=2)
        overall_alpha_lb = overall_alpha_lb.clip(min=0.0, max=0.99)
        overall_alpha_ub = overall_alpha_ub.clip(min=0.0, max=0.99)

        inp_depth = torch.clone(cam_inp).repeat(BS, 1)
        # print(">>>>>> Starting Bounded Module")
        model_depth_bounded = BoundedModule(model_depth, (inp_depth, means_hom_tmp), device=means_strip.device)
        # print(">>>>>> Starting PerturbationLpNorm")
        ptb_depth = PerturbationLpNorm(norm=np.inf, x_L=inp_depth+eps_lb, x_U=inp_depth+eps_ub)
        # print(">>>>>> Starting BoundedTensor")
        inp_depth = BoundedTensor(inp_depth, ptb_depth)
        prediction = model_depth_bounded(inp_depth, means_hom_tmp)
        # model_depth_bounded.visualize('depth')
        required_A = defaultdict(set)
        required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
        lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, means_hom_tmp), method=method, return_A=True, needed_A_dict=required_A)

        depth_lA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lA'].transpose(0,1)
        depth_uA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['uA'].transpose(0,1)
        depth_lbias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lbias'].transpose(0,1)
        depth_ubias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['ubias'].transpose(0,1)

        overall_depth_lA = torch.cat((overall_depth_lA, depth_lA), dim=1)
        overall_depth_uA = torch.cat((overall_depth_uA, depth_uA), dim=1)
        overall_depth_lbias = torch.cat((overall_depth_lbias, depth_lbias), dim=1)
        overall_depth_ubias = torch.cat((overall_depth_ubias, depth_ubias), dim=1)

    bounds_alpha = torch.cat((overall_alpha_lb, overall_alpha_ub), dim=0)
    nan_mask = torch.any(torch.isnan(bounds_alpha),dim=(0,1,3))
    inf_mask = torch.any(torch.isinf(bounds_alpha),dim=(0,1,3))
    mask = ~(nan_mask | inf_mask) 
    bounds_alpha = bounds_alpha[:,:,mask,:]
    overall_depth_lA = overall_depth_lA[:,mask,:]
    overall_depth_uA = overall_depth_uA[:,mask,:]
    overall_depth_lbias = overall_depth_lbias[:,mask]
    overall_depth_ubias = overall_depth_ubias[:,mask]
    colors_strip = colors_strip[mask,:]

    tmp_res_Tl, tmp_res_Tu = computeT_new_new(
        ptb_depth,
        overall_depth_lA,
        overall_depth_uA,
        overall_depth_lbias,
        overall_depth_ubias,   
        bounds_alpha[0:1],
        bounds_alpha[1:2]
    )

    res_T = torch.cat((tmp_res_Tl, tmp_res_Tu), dim=0)
    bounds_colors = torch.stack((colors_strip, colors_strip), dim=0)
    bounds_colors = bounds_colors[:,None]
    tile_color = (res_T*bounds_alpha*bounds_colors).sum(dim=2)
    tile_color_lb = tile_color[0]
    tile_color_ub = tile_color[1]
    tile_color_lb = tile_color_lb.reshape(htr-hbl, wtr-wbl, -1)[:,:,:3]
    tile_color_ub = tile_color_ub.reshape(htr-hbl, wtr-wbl, -1)[:,:,:3]
    tile_color_lb = tile_color_lb.detach().cpu().numpy()
    tile_color_ub = tile_color_ub.detach().cpu().numpy()        
    return tile_color_lb, tile_color_ub

if __name__ == "__main__":
    transform = np.array(dt['transform'])
    transform_ap = np.vstack((transform, np.array([0,0,0,1])))
    scale = dt['scale']

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/triangular_osm/splatfacto/2025-01-23_021826')
    checkpoint = "step-000059999.ckpt"
    
    checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    res = torch.load(checkpoint_fn)
    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']
    colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

    color_mask = torch.norm(colors, dim=1)>0.1
    means = means[color_mask]
    quats = quats[color_mask]
    opacities = opacities[color_mask]
    scales = scales[color_mask]
    colors = colors[color_mask]

    camera_pose = np.array([
        [
            0.0,
            0.0,
            -1.0,
            -160.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            10.0,
        ],
        [
            1.0,
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
    pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to(means.device)
    # Get the rectangles of gaussians under uncertainty 
    rect, mask = get_rect_set(
        cam_inp+eps_lb,
        cam_inp+eps_ub,
        means,
        scales,
        quats,
        f,
        f,
        width,
        height 
    )
    render_color_lb = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
    render_color_ub = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
    means = means[mask]
    quats = quats[mask]
    opacities = opacities[mask]
    scales = scales[mask]
    colors = colors[mask]
    
    if adaptive_sampling:
        queue = [
            (h,w,min(h+initial_tilesize, height),min(w+initial_tilesize, width), initial_tilesize) \
            for h in range(0, height, initial_tilesize) for w in range(0, width, initial_tilesize) 
        ]
    else:
        queue = [
            (h,w,min(h+tile_size_global, height),min(w+tile_size_global, width), tile_size_global) \
            for h in range(0, height, tile_size_global) for w in range(0, width, tile_size_global) 
        ] 
    # Implement adaptive tile size 
    while queue!=[]:
        hbl,wbl,htr,wtr,tile_size = queue[0]
        queue.pop(0)
        over_tl = rect[0][..., 0].clip(min=wbl), rect[0][..., 1].clip(min=hbl)
        over_br = rect[1][..., 0].clip(max=wtr-1), rect[1][..., 1].clip(max=htr-1)
        in_mask = (over_br[0] >= over_tl[0]) & (over_br[1] >= over_tl[1])
        if not in_mask.sum() > 0:
            continue
        N = torch.where(in_mask)[0].shape[0]
        # If tile size too large or too much gaussians 
        if tile_size**2*N>threshold and tile_size>tile_size_global:
            if tile_size == 1:
                raise ValueError(f"Tile size can't be partitioned anymore, too many gaussians to be handled for ({hbl}, {wbl}), ({htr}, {wtr})")
            tile_size = tile_size//2 
            new_partitions = [
                (h,w,min(h+tile_size, htr),min(w+tile_size, wtr), tile_size) \
                for w in range(wbl, wtr, tile_size) for h in range(hbl, htr, tile_size)
            ]
            queue = queue+new_partitions 
            continue 
        means_strip = means[in_mask]
        quats_strip = quats[in_mask]
        opacities_strip = opacities[in_mask]
        scales_strip = scales[in_mask]
        colors_strip = colors[in_mask]
        print(f">>>>>>>> {hbl}, {wbl}, {htr}, {wtr}, {means_strip.shape[0]}")

        tile_color_lb, tile_color_ub = compute_tile_color(
            cam_inp,
            eps_lb,
            eps_ub,
            means_strip,
            opacities_strip,
            scales_strip,
            quats_strip,
            colors_strip,
            f,
            wbl,
            hbl,
            wtr,
            htr,
            width,
            height,
            pix_coord,
            tile_size,
            gauss_step
        )
        render_color_lb[hbl:htr, wbl:wtr] = tile_color_lb
        render_color_ub[hbl:htr, wbl:wtr] = tile_color_ub
        plt.imshow(render_color_lb)
        plt.savefig('res_lb.png')
        plt.imshow(render_color_ub)
        plt.savefig('res_ub.png')
        

    # for h in range(0, height, tile_size_global):
    #     for w in range(0, width, tile_size_global):
    #         # if h!=8 or w!=0:
    #         #     continue
    #         # if h>24:
    #         #     continue
    #         over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
    #         over_br = rect[1][..., 0].clip(max=w+tile_size_global-1), rect[1][..., 1].clip(max=h+tile_size_global-1)
    #         in_mask = (over_br[0] >= over_tl[0]) & (over_br[1] >= over_tl[1])
    #         if not in_mask.sum() > 0:
    #             continue
    #         means_strip = means[in_mask]
    #         quats_strip = quats[in_mask]
    #         opacities_strip = opacities[in_mask]
    #         scales_strip = scales[in_mask]
    #         colors_strip = colors[in_mask]
    #         print(f">>>>>>>> {h}, {w}, {means_strip.shape[0]}")

    #         tile_color_lb, tile_color_ub = compute_tile_color(
    #             cam_inp,
    #             eps_lb,
    #             eps_ub,
    #             means_strip,
    #             opacities_strip,
    #             scales_strip,
    #             quats_strip,
    #             colors_strip,
    #             f,
    #             w,
    #             h,
    #             w+tile_size_global,
    #             h+tile_size_global,
    #             width,
    #             height,
    #             pix_coord,
    #             tile_size_global,
    #             gauss_step
    #         )
    #         render_color_lb[h:h+tile_size_global, w:w+tile_size_global] = tile_color_lb
    #         render_color_ub[h:h+tile_size_global, w:w+tile_size_global] = tile_color_ub
    #         plt.imshow(render_color_lb)
    #         plt.savefig('res_lb.png')
    #         plt.imshow(render_color_ub)
    #         plt.savefig('res_ub.png')
    # plt.figure(1)
    # plt.imshow(render_color_lb)
    # plt.figure(2)
    # plt.imshow(render_color_ub)
    # plt.show()
