from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
# import os 
import numpy as np 
import matplotlib.pyplot as plt 
from simple_model2_alphatest4 import AlphaModel, DepthModel, MeanModel
from scipy.spatial.transform import Rotation 
from collections import defaultdict

method = 'alpha-crown'
adaptive_sampling = False 

width = 20
height = 20 
f = width*2

eps_lb = torch.Tensor([[0, 0.0, 0, -1.0, -1.0, -1.0]]).to('cuda')
eps_ub = torch.Tensor([[0,  0.0, 0,  1.0,  1.0,  1.0]]).to('cuda')
tile_size_global = 16
if method == 'ibp' or method=='crown':
    gauss_step = 2000
elif method == 'alpha-crown':
    gauss_step = 1500
threshold = tile_size_global**2*gauss_step
initial_tilesize = 16

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
    means_proj_hom = model_mean(cam_inp, means_hom_tmp)
    means2D = (means_proj_hom[:,:,:2]/means_proj_hom[:,:,2:]).squeeze()

    radii = model_mean.get_radii(cam_inp)
    radii = radii.squeeze()
    rect_min = means2D-radii[:,None] 
    rect_max = means2D+radii[:,None]
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

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
    # mask = (depth>0.05) & (depth<10000000000)
    # means_hom_tmp = 

    means_hom_tmp = model_mean.means_hom_tmp.transpose(0,2)
    model_mean_bounded = BoundedModule(model_mean, (inp_mean, means_hom_tmp), device = means.device)
    ptb_mean = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
    inp_mean = BoundedTensor(inp_mean, ptb_mean)
    lb_mean, ub_mean = model_mean_bounded.compute_bounds(x=(inp_mean, means_hom_tmp), method='ibp')

    means_proj_hom = model_mean((x_L+x_U)/2, means_hom_tmp)

    lb_mean = lb_mean.squeeze()
    ub_mean = ub_mean.squeeze()
    mask = (lb_mean[:,2]>0.01) & (ub_mean[:,2]<10000000000)
    lb_mean = lb_mean[mask]
    ub_mean = ub_mean[mask]
    # lb_mean = lb_mean.clip(min=1e-8)
    # ub_mean = ub_mean.clip(min=1e-8)
    lb_mean2D = lb_mean[:,:2]/ub_mean[:,2:]
    ub_mean2D = ub_mean[:,:2]/lb_mean[:,2:]
    
    means2D = (means_proj_hom[:,mask,:2]/means_proj_hom[:,mask,2:]).squeeze()

    radii = model_mean.get_radii((x_L+x_U)/2)
    radii = radii.squeeze()[mask]
    rect_min = lb_mean2D-radii[:,None]
    rect_max = ub_mean2D+radii[:,None]
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    rect_min_tmp = means2D-radii[:,None]
    rect_max_tmp = means2D-radii[:,None]
    rect_min_tmp[..., 0] = rect_min_tmp[..., 0].clip(0, width - 1.0)
    rect_min_tmp[..., 1] = rect_min_tmp[..., 1].clip(0, height - 1.0)
    rect_max_tmp[..., 0] = rect_max_tmp[..., 0].clip(0, width - 1.0)
    rect_max_tmp[..., 1] = rect_max_tmp[..., 1].clip(0, height - 1.0)
    return (rect_min, rect_max), (rect_min_tmp, rect_max_tmp), mask 

dt = {
    "transform": [
        [
            0.9971602559089661,
            -0.07308899611234665,
            -0.018150581046938896,
            0.0732671394944191
        ],
        [
            -0.018150581046938896,
            -0.467155784368515,
            0.8839886784553528,
            -0.22318288683891296
        ],
        [
            -0.07308899611234665,
            -0.8811489343643188,
            -0.467155784368515,
            0.17995375394821167
        ]
    ],
    "scale": 0.1839942726392184
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
    # wbl,
    # hbl,
    # wtr,
    # htr,
    width,
    height,

    # Add parameters
    pix_coord,
    # tile_size,
    # gauss_step,
):
    tile_coord = pix_coord.flatten(0,-2)
    # N = means_strip.shape[0]

    # overall_alpha_lb = torch.zeros((1,(htr-hbl)*(wtr-wbl), 0, 1)).to(means_strip.device)
    # overall_alpha_ub = torch.zeros((1,(htr-hbl)*(wtr-wbl), 0, 1)).to(means_strip.device)

    # overall_depth_lA = torch.zeros((1,0,6)).to(means_strip.device)
    # overall_depth_uA = torch.zeros((1,0,6)).to(means_strip.device)
    # overall_depth_lbias = torch.zeros((1,0)).to(means_strip.device)
    # overall_depth_ubias = torch.zeros((1,0)).to(means_strip.device)
    
    # for j in range(0, N, gauss_step):
    # print(f">>>>>>>> Computation Progress {j}/{N}")
    data_pack = {
        'opacities': torch.Tensor(opacities_strip),
        'means': torch.Tensor(means_strip),
        'scales':torch.Tensor(scales_strip),
        'quats':torch.Tensor(quats_strip),
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

    model_depth = DepthModel(model_alpha)

    inp_alpha = torch.clone(cam_inp).repeat(BS, 1)
    model_alpha_bounded = BoundedModule(
        model_alpha, 
        (inp_alpha, means_hom_tmp, cov_world, opacities_rast), 
        device=means_strip.device,
        bound_opts= {
            'conv_mode': 'matrix',
            'optimize_bound_args': {
                # 'iteration': 100, 
                'early_stop_patience':5},
        }, 
    )
    ptb_alpha = PerturbationLpNorm(norm=np.inf, x_L=inp_alpha+eps_lb, x_U=inp_alpha+eps_ub)
    inp_alpha = BoundedTensor(inp_alpha, ptb_alpha)
    tmp_lb_alpha, tmp_ub_alpha = model_alpha_bounded.compute_bounds(
        x=(inp_alpha, means_hom_tmp, cov_world, opacities_rast), 
        method=method
    )
    tmp_lb_alpha2 = torch.minimum(tmp_lb_alpha, tmp_ub_alpha)
    tmp_ub_alpha2 = torch.maximum(tmp_lb_alpha, tmp_ub_alpha)
    lb_alpha = -tmp_ub_alpha2*0.5
    ub_alpha = -tmp_lb_alpha2*0.5
    lb_alpha = torch.exp(lb_alpha)
    ub_alpha = torch.exp(ub_alpha)
    ub_alpha = ub_alpha.clip(min = 0.0, max=1.0)
    lb_alpha = lb_alpha.clip(min = 0.0, max=1.0)
    lb_alpha = lb_alpha*opacities_rast
    ub_alpha = ub_alpha*opacities_rast
    lb_alpha = lb_alpha.transpose(0,1)[None,:,:,None]
    ub_alpha = ub_alpha.transpose(0,1)[None,:,:,None]
    overall_alpha_lb = lb_alpha
    overall_alpha_ub = ub_alpha
    overall_alpha_lb = overall_alpha_lb.clip(min=0.0, max=0.99)
    overall_alpha_ub = overall_alpha_ub.clip(min=0.0, max=0.99)

    emp_alpha_lb = torch.zeros(overall_alpha_lb.shape,device=overall_alpha_lb.device)+1e10
    emp_alpha_ub = torch.zeros(overall_alpha_ub.shape,device=overall_alpha_ub.device)-1e10
    for i in range(1000):
        random_values = torch.rand_like(eps_lb)
        cam_inp_tmp = cam_inp+torch.lerp(eps_lb, eps_ub, random_values)
        inp_alpha = cam_inp_tmp.repeat((means_hom_tmp.shape[0],1))
        alpha_res = model_alpha(inp_alpha, means_hom_tmp, cov_world, opacities_rast)
        alpha_res = alpha_res*(-0.5)
        alpha_res = torch.exp(alpha_res)
        alpha_res = alpha_res*opacities_rast
        alpha_res = alpha_res.transpose(0,1)[None,:,:,None]
        emp_alpha_lb = torch.minimum(alpha_res,emp_alpha_lb)
        emp_alpha_ub = torch.maximum(alpha_res,emp_alpha_ub)
    diff_lb = emp_alpha_lb-overall_alpha_lb
    diff_ub = overall_alpha_ub-emp_alpha_ub

    inp_depth = torch.clone(cam_inp).repeat(BS, 1)
    model_depth_bounded = BoundedModule(model_depth, (inp_depth, means_hom_tmp), device=means_strip.device)
    ptb_depth = PerturbationLpNorm(norm=np.inf, x_L=inp_depth+eps_lb, x_U=inp_depth+eps_ub)
    inp_depth = BoundedTensor(inp_depth, ptb_depth)
    prediction = model_depth_bounded(inp_depth, means_hom_tmp)
    required_A = defaultdict(set)
    required_A[model_depth_bounded.output_name[0]].add(model_depth_bounded.input_name[0])
    lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, means_hom_tmp), method=method, return_A=True, needed_A_dict=required_A)

    depth_lA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lA'].transpose(0,1)
    depth_uA: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['uA'].transpose(0,1)
    depth_lbias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['lbias'].transpose(0,1)
    depth_ubias: torch.Tensor = A_depth[model_depth_bounded.output_name[0]][model_depth_bounded.input_name[0]]['ubias'].transpose(0,1)

    overall_depth_lA = depth_lA
    overall_depth_uA = depth_uA
    overall_depth_lbias = depth_lbias
    overall_depth_ubias = depth_ubias

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
    tile_color_lb = tile_color_lb.reshape(height, width, -1)[:,:,:3]
    tile_color_ub = tile_color_ub.reshape(height, width, -1)[:,:,:3]
    tile_color_lb = tile_color_lb.detach().cpu().numpy()
    tile_color_ub = tile_color_ub.detach().cpu().numpy()        
    return tile_color_lb, tile_color_ub

if __name__ == "__main__":
    transform = np.array(dt['transform'])
    transform_ap = np.vstack((transform, np.array([0,0,0,1])))
    scale = dt['scale']

    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/dozer/splatfacto/2025-01-27_164355')
    # checkpoint = "step-000029999.ckpt"
    
    # checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    # res = torch.load(checkpoint_fn)
    # means = res['pipeline']['_model.gauss_params.means']
    # quats = res['pipeline']['_model.gauss_params.quats']
    # opacities = res['pipeline']['_model.gauss_params.opacities']
    # scales = res['pipeline']['_model.gauss_params.scales']
    # colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

    means = torch.Tensor(np.array([
        [-2,0, 10],
        [0,0, 10.5],
        [2,0, 11]
    ])).to('cuda')
    # Orientations of three gaussian
    # rpys = np.random.uniform([-np.pi/2,-np.pi/2,-np.pi/2], [np.pi/2,np.pi/2,np.pi/2], (1,N,3))
    rpys = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ])
    quats = Rotation.from_euler('xyz', rpys).as_quat() # x,y,z,w
    quats = np.hstack((quats[:,3:4], quats[:,0:1], quats[:,1:2], quats[:,2:3]))
    quats = torch.Tensor(quats).to('cuda')
    
    # Scales of three gaussian, all scales between -1-0
    # scales = np.random.uniform(-1, -0.2, (1,N,3))
    scales = torch.Tensor(np.array([
        [-0.4,-0.2,-0.4],
        [-0.2,-0.2,-0.4],
        [-0.2,-0.4,-0.4]
    ])).to('cuda')
    # Setup Opacities of three gaussian, all between 0.5-0.8 (relatively opaque)
    # opacities = np.random.uniform(0.5, 0.8, (1,N,1))
    opacities = torch.Tensor(np.array([
        [0.6],
        [0.6],
        [0.6]
    ])).to('cuda')

    colors = torch.Tensor(np.array([
        [1.0,1.0,0.0],
        [1.0,0.0,1.0],
        [0.0,1.0,1.0]
    ])).to('cuda')

    # mask_opacities = torch.sigmoid(opacities).squeeze()>0.15
    # mask_scale = torch.all(scales>-8.0, dim=1)
    # means = means[mask_scale&mask_opacities]
    # quats = quats[mask_scale&mask_opacities]
    # opacities = opacities[mask_scale&mask_opacities]
    # scales = scales[mask_scale&mask_opacities]
    # colors = colors[mask_scale&mask_opacities]

    # means_tmp = torch.cat((means, torch.ones(means.shape[0],1).to('cuda')), dim=1)
    # means_w = torch.inverse(torch.Tensor(transform_ap).to('cuda'))@means_tmp.transpose(0,1)/scale

    Ks = torch.tensor([[
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]
    ]]).to(torch.device('cuda'))

    cam_inp = torch.Tensor(np.array([[
        0,0,0,0,0,0
    ]])).to('cuda')
    
    pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to(means.device)
    tile_coord = pix_coord.flatten(0,-2)
    data_pack = {
        'opacities': torch.Tensor(opacities),
        'means': torch.Tensor(means),
        'scales':torch.Tensor(scales),
        'quats':torch.Tensor(quats),
    } 

    model_alpha = AlphaModel(
        data_pack, f, f, width, height, tile_coord
    )
    model_depth = DepthModel(model_alpha)
    means_hom_tmp = model_alpha.means_hom_tmp.transpose(0,2)
    cov_world = model_alpha.cov_world 
    opacities_rast = model_alpha.opacities_rast.transpose(0,2)[:,:,0,0]
    
    inp_alpha = cam_inp.repeat((means_hom_tmp.shape[0],1))
    inp_depth = cam_inp.repeat((means_hom_tmp.shape[0],1))
    alpha_res = model_alpha(inp_alpha, means_hom_tmp, cov_world, opacities_rast)
    depth_res = model_depth(inp_depth, means_hom_tmp)

    alpha_res = alpha_res*(-0.5)
    alpha_res = torch.exp(alpha_res)
    alpha_res = alpha_res*opacities_rast

    overall_alpha = alpha_res.transpose(0,1)[None,:,:,None]
    overall_depth = depth_res.transpose(0,1)

    depth_order = torch.argsort(overall_depth, dim=1).squeeze()
    sorted_alpha = overall_alpha[0,:,depth_order,:]
    sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:1]), 1-sorted_alpha[:,:-1]], dim=1).cumprod(dim=1)
    sorted_color = colors[depth_order,:]
    rgb_color = (sorted_T * sorted_alpha * sorted_color[None]).sum(dim=1)
    rgb_color = rgb_color.reshape((height, width, -1))[:,:,:3]
    rgb_color = rgb_color.detach().cpu().numpy()
    plt.imshow(rgb_color)
    plt.show()
    
    render_color_lb = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
    render_color_ub = np.zeros((*pix_coord.shape[:2], colors.shape[-1]))
    
    # queue = [
    #     (h,w,min(h+tile_size_global, height),min(w+tile_size_global, width), tile_size_global) \
    #     for h in range(0, height, tile_size_global) for w in range(0, width, tile_size_global) 
    # ] 
    # Implement adaptive tile size 
    # while queue!=[]:
    # hbl,wbl,htr,wtr,tile_size = queue[0]
    # queue.pop(0)
    # if (hbl!=48 or wbl!=64):
    #     continue
    # print(f">>>>>>>> {hbl}, {wbl}, {htr}, {wtr}, {means.shape[0]}")

    tile_color_lb, tile_color_ub = compute_tile_color(
        cam_inp,
        eps_lb,
        eps_ub,
        means,
        opacities,
        scales,
        quats,
        colors,
        f,
        # wbl,
        # hbl,
        # wtr,
        # htr,
        width,
        height,
        pix_coord,
        # tile_size,
        # gauss_step
    )
    render_color_lb = tile_color_lb
    render_color_ub = tile_color_ub
    plt.figure(0)
    plt.imshow(render_color_lb)
    plt.savefig('res_lb.png')
    plt.figure(1)
    plt.imshow(render_color_ub)
    plt.savefig('res_ub.png')
    plt.show()
        
    # from PIL import Image 
    # render_color_lb = (render_color_lb.clip(min=0.0, max=1.0)*255).astype(np.uint8)
    # render_color_ub = (render_color_ub.clip(min=0.0, max=1.0)*255).astype(np.uint8)
    # res_lb = Image.fromarray(render_color_lb)
    # res_ub = Image.fromarray(render_color_ub)
    # new_width = width*5
    # new_height = height*5 
    # res_lb_enlarged = res_lb.resize((new_width, new_height), Image.NEAREST)
    # res_ub_enlarged = res_ub.resize((new_width, new_height), Image.NEAREST)
    # res_lb_enlarged.save('dozer1_good_lb.png')
    # res_ub_enlarged.save('dozer1_good_ub.png')

    # print(f">>>>>>> {eps_lb}, {eps_ub}, mean_diff {np.mean(np.linalg.norm(render_color_ub-render_color_lb,axis=2))}, max_diff {np.max(np.linalg.norm(render_color_ub-render_color_lb,axis=2))}")
    
