from simple_model2_alphatest2 import MeanModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import numpy as np 
import itertools 
from typing import Dict 

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
    means_hom_tmp = model_mean.means_hom_tmp
    means_proj_hom = model_mean(means_hom_tmp, cam_inp)
    means2D = (means_proj_hom[:,:,:2]/means_proj_hom[:,:,2:]).squeeze()

    radii = model_mean.get_radii(means_hom_tmp, cam_inp)
    radii = radii.squeeze()
    rect_min = means2D-radii[:,None] 
    rect_max = means2D+radii[:,None]
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

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
    means_strip = means#[:10000]
    scales_strip = scales#[:10000]
    quats_strip = quats#[:10000]
    
    lb_mean = torch.zeros((1,0,3)).to(means_strip.device)
    ub_mean = torch.zeros((1,0,3)).to(means_strip.device)
    N = means_strip.shape[0]

    # BS = 5000

    # for i in range(0,N,BS):
    inp_mean = torch.clone((x_L+x_U)/2)
    model_mean = MeanModel(means_strip, scales_strip, quats_strip, fx, fy, width, height)
    model_mean_bounded = BoundedModule(model_mean, (inp_mean, ), device = means.device)
    ptb_mean = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
    inp_mean = BoundedTensor(inp_mean, ptb_mean)
    lb_mean_part, ub_mean_part = model_mean_bounded.compute_bounds(x=(inp_mean, ), method='ibp')
    lb_mean = torch.cat((lb_mean, lb_mean_part), dim=1)
    ub_mean = torch.cat((ub_mean, ub_mean_part), dim=1)

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

def linear_bounds(A,b,x_L, x_U):
    pos_mask = (A>=0).float() 
    neg_mask = 1.0-pos_mask 

    A_pos = A*pos_mask 
    A_neg = A*neg_mask 

    fmin = torch.einsum('iabc,ic->iab',A_pos,x_L)+torch.einsum('iabc,ic->iab',A_neg,x_U)+b 
    fmax = torch.einsum('iabc,ic->iab',A_pos,x_U)+torch.einsum('iabc,ic->iab',A_neg,x_L)+b
    return fmin, fmax

def get_bound_depth_step(
    ptb: PerturbationLpNorm,
    lA: torch.Tensor,
    uA: torch.Tensor,
    lbias: torch.Tensor,
    ubias: torch.Tensor,
):
    x_L = ptb.x_L
    x_U = ptb.x_U
    
    lA_diff = lA[:,:,None]-uA[:,None,:]
    uA_diff = uA[:,:,None]-lA[:,None,:]
    lbias_diff = lbias[:,:,None]-ubias[:,None,:]
    ubias_diff = ubias[:,:,None]-lbias[:,None,:]

    diffL_part, _ = linear_bounds(lA_diff, lbias_diff, x_L, x_U)
    _, diffU_part = linear_bounds(uA_diff, ubias_diff, x_L, x_U)
    diffL = torch.minimum(diffL_part, diffU_part)
    diffU = torch.maximum(diffL_part, diffU_part)

    mask = torch.ones(diffL.shape[1], diffL.shape[1], device=diffL.device)
    mask.fill_diagonal_(0)
    diffL = diffL*mask[None]
    diffU = diffU*mask[None]

    assert torch.all(diffL<=diffU)

    step_L = torch.zeros(diffL.shape)
    mask_L = diffL>0
    step_L[mask_L] = 1.0 
    step_U = torch.ones(diffU.shape)
    mask_U = diffU<=0
    step_U[mask_U] = 0.0

    return step_L, step_U

def get_elem_before_linear(
    ptb: PerturbationLpNorm, 
    lA: torch.Tensor, 
    uA: torch.Tensor, 
    lbias: torch.Tensor, 
    ubias: torch.Tensor, 
):
    x_L = ptb.x_L
    x_U = ptb.x_U
    x_bounds = torch.cat((x_L, x_U), dim=0)
    x_bounds_list = x_bounds.transpose(0,1).detach().cpu().numpy().tolist()
    all_combins = list(itertools.product(*x_bounds_list))
    all_combins = torch.Tensor(all_combins).transpose(0,1).to(lA.device)

    concrete_bound = {}
    possible_bound = {}
    for i in range(lA.shape[1]):
        concrete_bound[i] = []
        possible_bound[i] = []
        for j in range(lA.shape[1]):
            if i==j:
                continue
            fxl = lA[:,j,:]@all_combins+lbias[:,j]
            fxu = uA[:,j,:]@all_combins+ubias[:,j]
            fxl_ref = lA[:,i,:]@all_combins+lbias[:,i]
            fxu_ref = uA[:,i,:]@all_combins+ubias[:,i]
            if torch.all(fxu<=fxl_ref):
                concrete_bound[i].append(j)
                possible_bound[i].append(j) 
            elif torch.all(fxl>=fxu_ref):
                pass 
            else:
                possible_bound[i].append(j)
            # for k in range(len(all_combins)):
            #     x = torch.Tensor(all_combins[k]).to(lA.device)
            #     fxl = lA[0,j,:]@x+lbias[0,j]
            #     fxu = uA[0,j,:]@x+ubias[0,j]
            #     fxl_ref = lA[0,i,:]@x+lbias[0,i]
            #     fxu_ref = uA[0,i,:]@x+ubias[0,i]
            #     if fxu<=fxl_ref:
            #         smaller_than = smaller_than or True 
            #     elif fxl>=fxu_ref:
            #         greater_than = greater_than or True
            # if smaller_than and not greater_than:
            #     concrete_bound[i].append(j)
            #     possible_bound[i].append(j)
            # elif smaller_than and greater_than:
            #     possible_bound[i].append(j)
            # elif not smaller_than and not greater_than:
            #     possible_bound[i].append(j)
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

def computeT_new(alpha_bound: torch.Tensor, step_res: torch.Tensor):
    T = torch.zeros(alpha_bound.shape).to(alpha_bound.device)
    # alpha_bound_cpu = alpha_bound.to('cpu')
    for i in range(step_res.shape[2]):
        T[0,:,i,:] = (torch.ones((1,1,1,1)).to(alpha_bound.device)-(alpha_bound*step_res[:,None,i,:,None].to(alpha_bound.device))).prod(dim=2)
    return T.to('cuda') 

def computeT_new_optimized(alpha_bound: torch.Tensor, step_res: torch.Tensor) -> torch.Tensor:
    # Remove singleton dimensions (batch and channel)
    A = alpha_bound.squeeze(0).squeeze(-1)  # Shape: (P, N)
    device = A.device
    S = step_res.squeeze(0).to(device)                 # Shape: (N, N)
    P, N = A.shape[0], S.shape[0]
    
    # Initialize result as ones (P, N)
    result = torch.ones((P, N), device=device)
    
    # Iterate over j to compute the product incrementally
    for j in range(N):
        A_j = A[:, j]          # Shape: (P,)
        S_j = S[:, j]          # Shape: (N,)
        term = 1 - torch.outer(A_j, S_j)  # Shape: (P, N)
        result *= term
    
    # Restore original dimensions: (1, P, N, 1)
    return result.unsqueeze(0).unsqueeze(-1)

# def computeT_new_new(
#     ptb_depth: PerturbationLpNorm,
#     lA_depth: torch.Tensor,
#     uA_depth: torch.Tensor,
#     lbias_depth: torch.Tensor,
#     ubias_depth: torch.Tensor,
#     alpha_lb: torch.Tensor,
#     alpha_ub: torch.Tensor,
# ):
#     batch_size = 100
#     N = lA_depth.shape[1]
#     x_L = ptb_depth.x_L
#     x_U = ptb_depth.x_U
#     A_lb = alpha_lb.squeeze(0).squeeze(-1)
#     A_ub = alpha_ub.squeeze(0).squeeze(-1)
#     device = A_lb.device
#     P = A_lb.shape[0]
#     result_lb = torch.ones((P,N), device = device)                      # P*N
#     result_ub = torch.ones((P,N), device = device)                      # P*N
#     for i in range(0,N,batch_size):  
#         # Get for each gaussian i for all gaussians, what step(d_i-d_j) should be 
#         lA_diff = lA_depth[:,:,None]-uA_depth[:,None,i:i+batch_size]             # 1*N*BS*6
#         uA_diff = uA_depth[:,:,None]-lA_depth[:,None,i:i+batch_size]             # 1*N*BS*6
#         lbias_diff = lbias_depth[:,:,None]-ubias_depth[:,None,i:i+batch_size]    # 1*N*BS
#         ubias_diff = ubias_depth[:,:,None]-lbias_depth[:,None,i:i+batch_size]    # 1*N*BS

#         diffL_part, _ = linear_bounds(lA_diff, lbias_diff, x_L, x_U)    # 1*N*BS
#         _, diffU_part = linear_bounds(uA_diff, ubias_diff, x_L, x_U)    # 1*N*BS

#         diffL = torch.minimum(diffL_part, diffU_part)                   # 1*N*BS
#         diffU = torch.maximum(diffL_part, diffU_part)                   # 1*N*BS

#         # Set diagonal element to 0
#         diffL[:,i,:] = 0
#         diffU[:,i,:] = 0

#         assert torch.all(diffL<=diffU)

#         step_L = torch.zeros(diffL.shape).to(A_ub.device)               # 1*N*BS
#         mask_L = diffL>0
#         step_L[mask_L] = 1.0
#         step_U = torch.ones(diffU.shape).to(A_ub.device)                # 1*N*BS
#         mask_U = diffU<=0
#         step_U[mask_U] = 0.0

#         term_lb = 1-step_U*A_ub[:,None,i:i+batch_size]                         # P*N
#         term_ub = 1-step_L*A_lb[:,None,i:i+batch_size]                         # P*N

#         result_lb = result_lb*term_lb.prod(dim=2)
#         result_ub = result_ub*term_ub.prod(dim=2)
#     return result_lb.unsqueeze(0).unsqueeze(-1), result_ub.unsqueeze(0).unsqueeze(-1)

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
    x_L = ptb_depth.x_L
    x_U = ptb_depth.x_U
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

        assert torch.all(diffL<=diffU)

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