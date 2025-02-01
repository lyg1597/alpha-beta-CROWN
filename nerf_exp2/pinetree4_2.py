from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from simple_model2_alphatest4 import AlphaModel, DepthModel, MeanModel
from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
from scipy.spatial.transform import Rotation 
from collections import defaultdict
# from img_helper import get_viewmat, \
#     get_bound_depth_step, \
#     computeT_new_optimized, \
#     computeT_new_new
# import time 

# class MyModel(torch.nn.Module):
#     def forward(self, x: torch.Tensor, step_res: torch.Tensor, colors: torch.Tensor):       
#         # x.shape = (P*BS)*N 
#         # step_res.shape = (P*BS)*N
#         # colors.shape = ?
#         # Compute 1-S*alpha
#         Tterm = 1-step_res*x 

#         # Compute ln(1-S*alpha)
#         lnTterm = torch.log(Tterm)

#         # Compute ln(1-S*alpha)+ln(alpha)
#         lnTterm = 

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

def computeT_new_new_new_new(
    ptb_depth: PerturbationLpNorm,
    lA_depth: torch.Tensor,
    uA_depth: torch.Tensor,
    lbias_depth: torch.Tensor,
    ubias_depth: torch.Tensor,
    alpha_lb: torch.Tensor,
    alpha_ub: torch.Tensor,    
    colors: torch.Tensor,
):
    batch_size = 5
    N = lA_depth.shape[1]
    depth_L = ptb_depth.x_L[0:1]
    depth_U = ptb_depth.x_U[0:1]
    alpha_lb = alpha_lb.squeeze(0).squeeze(-1)
    alpha_ub = alpha_ub.squeeze(0).squeeze(-1)
    device = alpha_lb.device
    P = alpha_lb.shape[0]

    A_result_lb = torch.zeros((P,3,N), device = device)     # P*3*N
    A_result_ub = torch.ones((P,3,N), device = device)      # P*3*N
    b_result_lb = torch.zeros((P,3), device = device)       # P*3
    b_result_ub = torch.zeros((P,3), device = device)       # P*3


    A_alpha_lb = torch.ones(alpha_lb.squeeze().shape,device=device)
    A_alpha_ub = torch.ones(alpha_ub.squeeze().shape,device=device)

    b_alpha_lb = torch.zeros(alpha_lb.squeeze().shape,device=device)
    b_alpha_ub = torch.zeros(alpha_ub.squeeze().shape,device=device)

    for i in range(0, N, batch_size):
        lA_diff = (lA_depth[:,None,i:i+batch_size]-uA_depth[:,:,None]).transpose(1,2)             # 1*BS*N*6
        uA_diff = (uA_depth[:,None,i:i+batch_size]-lA_depth[:,:,None]).transpose(1,2)             # 1*BS*N*6
        lbias_diff = (lbias_depth[:,None,i:i+batch_size]-ubias_depth[:,:,None]).transpose(1,2)    # 1*BS*N
        ubias_diff = (ubias_depth[:,None,i:i+batch_size]-lbias_depth[:,:,None]).transpose(1,2)    # 1*BS*N

        diffL_part, _ = linear_bounds(lA_diff, lbias_diff, depth_L, depth_U)    # 1*BS*N
        _, diffU_part = linear_bounds(uA_diff, ubias_diff, depth_L, depth_U)    # 1*BS*N

        diffL = torch.minimum(diffL_part, diffU_part)                   # 1*BS*N
        diffU = torch.maximum(diffL_part, diffU_part)                   # 1*BS*N

        actual_bs = min(batch_size, N - i)

        batch_indices = torch.arange(i, i + actual_bs, device=device)  # k values: i, i+1, ..., i+bs-1
        offset_indices = torch.arange(0, actual_bs, device=device)     # Positions within current batch

        diffL[:, offset_indices, batch_indices] = 0  # j=k masking
        diffU[:, offset_indices, batch_indices] = 0

        assert torch.all(diffL<=diffU)

        step_L = torch.zeros(diffL.shape).to(alpha_lb.device)               # 1*BS*N
        mask_L = diffL>0
        step_L[mask_L] = 1.0
        step_U = torch.ones(diffU.shape).to(alpha_ub.device)                # 1*BS*N
        mask_U = diffU<=0
        step_U[mask_U] = 0.0


    color_lb_lb, color_lb_ub = linear_bounds(A_result_lb[:,:,None], b_result_lb[:,:,None], alpha_lb, alpha_ub)
    color_ub_lb, color_ub_ub = linear_bounds(A_result_ub[:,:,None], b_result_ub[:,:,None], alpha_lb, alpha_ub)
    return color_lb_lb, color_ub_ub           


# def computeT_new_new_new(
#     ptb_depth: PerturbationLpNorm,
#     lA_depth: torch.Tensor,
#     uA_depth: torch.Tensor,
#     lbias_depth: torch.Tensor,
#     ubias_depth: torch.Tensor,
#     alpha_lb: torch.Tensor,
#     alpha_ub: torch.Tensor,    
#     colors: torch.Tensor,
# ):
#     batch_size = 5
#     N = lA_depth.shape[1]
#     depth_L = ptb_depth.x_L[0:1]
#     depth_U = ptb_depth.x_U[0:1]
#     alpha_lb = alpha_lb.squeeze(0).squeeze(-1)
#     alpha_ub = alpha_ub.squeeze(0).squeeze(-1)
#     device = alpha_lb.device
#     P = alpha_lb.shape[0]

#     A_result_lb = torch.zeros((P,3,N), device = device)     # P*3*N
#     A_result_ub = torch.ones((P,3,N), device = device)      # P*3*N
#     b_result_lb = torch.zeros((P,3), device = device)       # P*3
#     b_result_ub = torch.zeros((P,3), device = device)       # P*3


#     A_alpha_lb = torch.ones(alpha_lb.squeeze().shape,device=device)
#     A_alpha_ub = torch.ones(alpha_ub.squeeze().shape,device=device)

#     b_alpha_lb = torch.zeros(alpha_lb.squeeze().shape,device=device)
#     b_alpha_ub = torch.zeros(alpha_ub.squeeze().shape,device=device)

#     for i in range(0, N, batch_size):
#         lA_diff = (lA_depth[:,None,i:i+batch_size]-uA_depth[:,:,None]).transpose(1,2)             # 1*BS*N*6
#         uA_diff = (uA_depth[:,None,i:i+batch_size]-lA_depth[:,:,None]).transpose(1,2)             # 1*BS*N*6
#         lbias_diff = (lbias_depth[:,None,i:i+batch_size]-ubias_depth[:,:,None]).transpose(1,2)    # 1*BS*N
#         ubias_diff = (ubias_depth[:,None,i:i+batch_size]-lbias_depth[:,:,None]).transpose(1,2)    # 1*BS*N

#         diffL_part, _ = linear_bounds(lA_diff, lbias_diff, depth_L, depth_U)    # 1*BS*N
#         _, diffU_part = linear_bounds(uA_diff, ubias_diff, depth_L, depth_U)    # 1*BS*N

#         diffL = torch.minimum(diffL_part, diffU_part)                   # 1*BS*N
#         diffU = torch.maximum(diffL_part, diffU_part)                   # 1*BS*N

#         actual_bs = min(batch_size, N - i)

#         batch_indices = torch.arange(i, i + actual_bs, device=device)  # k values: i, i+1, ..., i+bs-1
#         offset_indices = torch.arange(0, actual_bs, device=device)     # Positions within current batch

#         diffL[:, offset_indices, batch_indices] = 0  # j=k masking
#         diffU[:, offset_indices, batch_indices] = 0

#         assert torch.all(diffL<=diffU)

#         step_L = torch.zeros(diffL.shape).to(alpha_lb.device)               # 1*BS*N
#         mask_L = diffL>0
#         step_L[mask_L] = 1.0
#         step_U = torch.ones(diffU.shape).to(alpha_ub.device)                # 1*BS*N
#         mask_U = diffU<=0
#         step_U[mask_U] = 0.0

#         # Lower & upper bound for step*alpha
#         A_alphastep_lb = A_alpha_lb[:,None,:]*step_L                        # P*BS*N
#         A_alphastep_ub = A_alpha_ub[:,None,:]*step_U                        # P*BS*N
#         b_alphastep_lb = b_alpha_lb[:,None,:]*step_L                        # P*BS*N
#         b_alphastep_ub = b_alpha_ub[:,None,:]*step_U                        # P*BS*N
        
#         # Concrete/Linear lower & upper bounds for 1-step*alpha
#         # Bound given as A_Ttermlb*alpha_lb[:,None,:]+b_Tterm_lb
#         A_Tterm_lb = -A_alphastep_ub                                        # P*BS*N
#         A_Tterm_ub = -A_alphastep_lb                                        # P*BS*N
#         b_Tterm_lb = 1-b_alphastep_ub                                       # P*BS*N
#         b_Tterm_ub = 1-b_alphastep_lb                                       # P*BS*N

#         Tterm_lb = 1-alpha_ub.squeeze()[:,None,:]*step_U                   # P*BS*N
#         Tterm_ub = 1-alpha_lb.squeeze()[:,None,:]*step_L                   # P*BS*N
        
#         y_lb_xl = A_Tterm_lb * alpha_lb[:,None,:] + b_Tterm_lb
#         y_lb_xu = A_Tterm_lb * alpha_ub[:,None,:] + b_Tterm_lb
#         y_lb_min = torch.where(A_Tterm_lb >= 0, y_lb_xl, y_lb_xu)  # (P, BS, 1)                        
#         y_ub_xl = A_Tterm_ub * alpha_lb[:,None,:] + b_Tterm_ub
#         y_ub_xu = A_Tterm_ub * alpha_ub[:,None,:] + b_Tterm_ub
#         y_ub_max = torch.where(A_Tterm_ub >= 0, y_ub_xu, y_ub_xl)  # (P, BS, 1)  

#         # Linear lower & upper bounds for ln(1-step*alpha), this also automatically give us \sum(ln(1-step*alpha))
#         # Bounds given as A_lnTterm_lb*alpha_lb[:,None,:]+b_Tterm_lb       
#         def compute_log_bounds_torch_final(A_lb, A_ub, b_lb, b_ub, x_l, x_u, eps=1e-16):
#             """
#             Computes near-optimal linear bounds for ln(y) using adaptive tangent points and second-order corrections.
            
#             Args:
#                 A_lb, A_ub: Tensors of shape (P, BS, N)
#                 b_lb, b_ub: Tensors of shape (P, BS, N)
#                 x_l, x_u: Tensors of shape (P, 1, N)
#                 eps: Threshold for small intervals
            
#             Returns:
#                 (A_lb_prime, b_lb_prime, A_ub_prime, b_ub_prime)
#             """
#             # =============================================
#             # Lower bound computation (unchanged, already optimal)
#             # =============================================
#             y_lb_xl = A_lb * x_l + b_lb
#             y_lb_xu = A_lb * x_u + b_lb
#             y_lb_min = torch.where(A_lb >= 0, y_lb_xl, y_lb_xu)
#             delta = x_u - x_l
#             delta_expanded = delta.expand_as(A_lb)
            
#             # Stabilized secant slope
#             log_ratio = torch.log(y_lb_xu.clamp(min=1e-16)) - torch.log(y_lb_xl.clamp(min=1e-16))
#             slope_secant = torch.zeros_like(A_lb)
#             valid_mask = (delta_expanded.abs() > eps)
#             slope_secant[valid_mask] = log_ratio[valid_mask] / delta_expanded[valid_mask]
            
#             # Second-order correction for small delta
#             y_lb_deriv = A_lb / y_lb_min.clamp(min=1e-16)
#             y_lb_second_deriv = -A_lb**2 / y_lb_min.clamp(min=1e-16)**2
#             slope_deriv = y_lb_deriv + 0.5 * y_lb_second_deriv * delta_expanded  # Taylor expansion
#             A_lb_prime = torch.where(delta_expanded.abs() < eps, slope_deriv, slope_secant)
#             b_lb_prime = torch.log(y_lb_min.clamp(min=1e-16)) - A_lb_prime * x_l.expand_as(A_lb)

#             # =============================================
#             # Tighter Upper bound using optimal tangent
#             # =============================================
#             # Optimal tangent point for y_ub
#             c_upper = torch.where(A_ub > 0, x_u, x_l)  # x_u if A_ub > 0, else x_l
#             y_ub_c = A_ub * c_upper + b_ub
            
#             # Stabilized slope with second-order correction
#             A_ub_prime = A_ub / y_ub_c.clamp(min=1e-16)
#             y_ub_second_deriv = -A_ub**2 / y_ub_c.clamp(min=1e-16)**2
#             delta_upper = (x_u - x_l).expand_as(A_ub)
#             A_ub_prime += 0.5 * y_ub_second_deriv * delta_upper  # Taylor correction
#             b_ub_prime = torch.log(y_ub_c.clamp(min=1e-16)) - A_ub_prime * c_upper

#             return A_lb_prime, b_lb_prime, A_ub_prime, b_ub_prime

#         A_lnTterm_lb, b_lnTterm_lb, A_lnTterm_ub, b_lnTterm_ub = compute_log_bounds_torch_final(
#             A_Tterm_lb, A_Tterm_ub, b_Tterm_lb, b_Tterm_ub, alpha_lb[:,None,:], alpha_ub[:,None,:]
#         )                                                                                               # P*BS*N
#         y_lb_xl = A_lnTterm_lb * alpha_lb[:,None,:] + b_lnTterm_lb
#         y_lb_xu = A_lnTterm_lb * alpha_ub[:,None,:] + b_lnTterm_lb
#         y_lb_min = torch.where(A_lnTterm_lb >= 0, y_lb_xl, y_lb_xu)  # (P, BS, 1)                        
#         y_ub_xl = A_lnTterm_ub * alpha_lb[:,None,:] + b_lnTterm_ub
#         y_ub_xu = A_lnTterm_ub * alpha_ub[:,None,:] + b_lnTterm_ub
#         y_ub_max = torch.where(A_lnTterm_ub >= 0, y_ub_xu, y_ub_xl)  # (P, BS, 1)   
#         # delta = (alpha_ub-alpha_lb)[:,None,:].repeat(1, batch_size, 1)                  # P*BS*N 
#         # epsilon = (Tterm_ub-Tterm_lb)/Tterm_lb.clamp(min=1e-16)                          # P*BS*N 
#         # log_ratio = torch.log(Tterm_ub.clamp(min=1e-16)) - torch.log(Tterm_ub.clamp(min=1e-16))
#         # slope_secant = torch.zeros_like(A_Tterm_lb, device=device)                      # P*BS*N
#         # valid_mask = (delta.abs() > 1e-8)                                               # P*BS*N
#         # slope_secant[valid_mask] = log_ratio[valid_mask] / delta[valid_mask]          # P*BS*N
#         # slope_deriv = (A_Tterm_lb / Tterm_lb.clamp(min=1e-16)) * (1 - 0.5 * epsilon)    # P*BS*N
#         # A_lnTterm_lb = torch.where(delta.abs() < 1e-8, slope_deriv, slope_secant)        # P*BS*N
#         # b_lnTterm_lb = torch.log(Tterm_lb.clamp(min=1e-16)) - A_lnTterm_lb * alpha_lb[:,None,:]

#         # # A_lnTterm_lb = (torch.log(A_Tterm_lb*alpha_ub[:,None,:]+b_Tterm_lb)-\
#         # #                 torch.log(A_Tterm_lb*alpha_lb[:,None,:]+b_Tterm_lb))/(alpha_ub[:,None,:]-alpha_lb[:,None,:])      # P*BS*N                                                  
#         # A_lnTterm_ub = A_Tterm_ub/(A_Tterm_ub*alpha_mid[:,None,:])+b_Tterm_ub                        # P*BS*N
#         # # b_lnTterm_lb = torch.log(A_Tterm_lb*alpha_lb[:,None,:]+b_Tterm_lb)-A_lnTterm_lb*alpha_lb[:,None,:]      # P*BS*N
#         # b_lnTterm_ub = torch.log(A_Tterm_ub*alpha_mid[:,None,:]+b_Tterm_ub)-A_lnTterm_ub*alpha_mid[:,None,:]    # P*BS*N

#         # Incoperating *alpha as sum(ln(1-step*alpha))+lnalpha
#         def compute_extreme_bounds(x_l, x_u, eps=1e-8, min_val=1e-16):
#             """
#             Computes bounds for ln(x) where x ∈ [x_l, x_u].
#             Args:
#                 x_l, x_u: Tensors of shape (P, BS) with x_l > 0, x_u > 0
#             Returns:
#                 (A_lb_prime, b_lb_prime, A_ub_prime, b_ub_prime)
#             """
#             x_l = x_l.clamp(min=min_val)
#             x_u = x_u.clamp(min=min_val)
#             x_l, x_u = torch.minimum(x_l, x_u), torch.maximum(x_l, x_u)

#             delta = x_u - x_l
            
#             # Lower bound (secant line between x_l and x_u)
#             log_ratio = torch.log(x_u) - torch.log(x_l)
#             A_lb_prime = torch.where(torch.abs(delta) > eps,
#                                 log_ratio / delta,
#                                 1/x_l)  # Derivative at x_l for tiny delta
#             b_lb_prime = torch.log(x_l) - A_lb_prime * x_l

#             # Upper bound (tangent at midpoint)
#             c = 0.5 * (x_l + x_u)
#             A_ub_prime = 1 / c
#             b_ub_prime = torch.log(c) - A_ub_prime * c

#             return A_lb_prime, b_lb_prime, A_ub_prime, b_ub_prime
        
#         A_lnalpha_lb, b_lnalpha_lb, A_lnalpha_ub, b_lnalpha_ub = compute_extreme_bounds(
#             alpha_lb[:,i:i+batch_size], alpha_ub[:,i:i+batch_size]
#         )                           # P*BS*1, P*BS*1, b_lnalpha_lb, b_lnalpha_lb
#         y_lb_xl = A_lnalpha_lb * alpha_lb[:,i:i+batch_size] + b_lnalpha_lb
#         y_lb_xu = A_lnalpha_lb * alpha_ub[:,i:i+batch_size] + b_lnalpha_lb
#         y_lb_min = torch.where(A_lnalpha_lb >= 0, y_lb_xl, y_lb_xu)  # (P, BS, 1)                        
#         y_ub_xl = A_lnalpha_ub * alpha_lb[:,i:i+batch_size] + b_lnalpha_ub
#         y_ub_xu = A_lnalpha_ub * alpha_ub[:,i:i+batch_size] + b_lnalpha_ub
#         y_ub_max = torch.where(A_lnalpha_ub >= 0, y_ub_xu, y_ub_xl)  # (P, BS, 1)                          
#         # A_lnalpha_lb = ((torch.log(alpha_ub[:,i:i+batch_size])-torch.log(alpha_lb[:,i:i+batch_size]))\
#         #     /(alpha_ub[:,i:i+batch_size]-alpha_lb[:,i:i+batch_size]))                                     # P*BS
#         # A_lnalpha_ub = (1/((alpha_ub[:,i:i+batch_size]+alpha_lb[:,i:i+batch_size])/2))                    # P*BS
#         # b_lnalpha_lb = (torch.log(alpha_lb[:,i:i+batch_size])-A_lnalpha_lb*alpha_lb[:,i:i+batch_size])    # P*BS
#         # b_lnalpha_ub = (torch.log((alpha_ub[:,i:i+batch_size]+alpha_lb[:,i:i+batch_size])/2)\
#         #     -A_lnalpha_ub*(alpha_ub[:,i:i+batch_size]+alpha_lb[:,i:i+batch_size])/2)                      # P*BS

#         A_lnalpha_lb = A_lnalpha_lb.unsqueeze(2)                                            # P*BS*1
#         A_lnalpha_ub = A_lnalpha_ub.unsqueeze(2)                                            # P*BS*1                                            
#         b_lnalpha_lb = b_lnalpha_lb.unsqueeze(2)                                            # P*BS*1
#         b_lnalpha_ub = b_lnalpha_ub.unsqueeze(2)                                            # P*BS*1

#         mask = torch.zeros(A_lnTterm_lb.shape,device=device)                                # P*BS*N
#         mask[:,offset_indices, batch_indices] = 1 

#         A_lnTterm_lb = A_lnTterm_lb+mask*A_lnalpha_lb                                       # P*BS*N
#         A_lnTterm_ub = A_lnTterm_ub+mask*A_lnalpha_ub                                       # P*BS*N
#         b_lnTterm_lb = b_lnTterm_lb+mask*b_lnalpha_lb                                       # P*BS*N
#         b_lnTterm_ub = b_lnTterm_ub+mask*b_lnalpha_ub                                       # P*BS*N

#         # Incoperating *color as sum(ln(1-step*alpha))+lnalpha+lncolor, make batchsize to P*BS*3*N
#         # Bounds given as (A_prod_lb*alpha_lb[:,None,None,:]).sum(dim=3)+b_prod_lb
#         A_prod_lb = A_lnTterm_lb[:,:,None,:].repeat(1,1,3,1)                            # P*BS*3*N, A is not influenced, just repeat
#         A_prod_ub = A_lnTterm_ub[:,:,None,:].repeat(1,1,3,1)                            # P*BS*3*N, A is not influenced, just repeat
#         # b_prod_lb = torch.einsum('pbn,bc->pbc',b_lnTterm_lb,colors[i:i+batch_size])     # P*BS*3, add color to b to form P*BS*3*N and add across N, shape of color: BS*3,
#         # b_prod_ub = torch.einsum('pbn,bc->pbc',b_lnTterm_ub,colors[i:i+batch_size])     # P*BS*3, add color to b to form P*BS*3*N and add across N, shape of color: BS*3,
#         b_prod_lb = b_lnTterm_lb.sum(dim=2)[:,:,None]+torch.log(colors[i:i+batch_size][None])
#         b_prod_ub = b_lnTterm_ub.sum(dim=2)[:,:,None]+torch.log(colors[i:i+batch_size][None])

#         prod_lb_lb, prod_lb_ub = linear_bounds(A_prod_lb, b_prod_lb, alpha_lb, alpha_ub)    # P*BS*3
#         prod_ub_lb, prod_ub_ub = linear_bounds(A_prod_ub, b_prod_ub, alpha_lb, alpha_ub)    # P*BS*3
#         prod_mid = (prod_ub_ub+prod_lb_lb)/2                                                # P*BS*3

#         # In this case, (sum(ln(1-step*alpha))+lnalpha+lncolor) is given by A_prod@x+b_prod
#         # A_prod is P*BS*3*N and x is P*N, A_prod@alpha should be P*BS*3
#         # The exponential is bounded as (A_exp_lb*alpha).sum(dim=3)+b_exp_lb with shape P*BS*3
#         # Now I need to apply exponential 

#         def compute_exp_bounds(A_lb, A_ub, b_lb, b_ub, x_l, x_u, eps=1e-8):
#             """
#             Inputs:
#                 A_lb: Tensor of shape (P, BS, 3, N)
#                 A_ub: Tensor of shape (P, BS, 3, N)
#                 b_lb: Tensor of shape (P, BS, 3)
#                 b_ub: Tensor of shape (P, BS, 3)
#                 x_l: Tensor of shape (P, 1, 1, N)  (lower bounds for x)
#                 x_u: Tensor of shape (P, 1, 1, N)  (upper bounds for x)
#                 eps: Threshold for numerical stability

#             Outputs:
#                 A_lb_prime: Lower bound coefficients for exp(y), shape (P, BS, 3, N)
#                 A_ub_prime: Upper bound coefficients for exp(y), shape (P, BS, 3, N)
#                 b_lb_prime: Lower bias terms, shape (P, BS, 3)
#                 b_ub_prime: Upper bias terms, shape (P, BS, 3)
#             """
#             device = A_lb.device
#             P, BS, _, N = A_lb.shape

#             # ----------------------------------------------
#             # Step 1: Compute bounds for z_low = A_lb*x + b_lb
#             # ----------------------------------------------
#             # For z_L_low (minimum of z_low)
#             mask_A_lb_neg = (A_lb < 0).to(device)  # Where A_lb is negative
#             x_min_for_z_low = torch.where(mask_A_lb_neg, x_u, x_l)  # (P, BS, 3, N)
#             z_L_low = (A_lb * x_min_for_z_low).sum(dim=-1) + b_lb  # (P, BS, 3)

#             # ----------------------------------------------
#             # Step 2: Compute bounds for z_up = A_ub*x + b_ub
#             # ----------------------------------------------
#             # For z_U_up (maximum of z_up)
#             mask_A_ub_pos = (A_ub >= 0).to(device)  # Where A_ub is positive
#             x_max_for_z_up = torch.where(mask_A_ub_pos, x_u, x_l)  # (P, BS, 3, N)
#             z_U_up = (A_ub * x_max_for_z_up).sum(dim=-1) + b_ub  # (P, BS, 3)

#             # For z_L_up (minimum of z_up)
#             x_min_for_z_up = torch.where(mask_A_ub_pos, x_l, x_u)  # (P, BS, 3, N)
#             z_L_up = (A_ub * x_min_for_z_up).sum(dim=-1) + b_ub  # (P, BS, 3)

#             # ----------------------------------------------
#             # Step 3: Compute stable upper bound coefficients
#             # ----------------------------------------------
#             # Avoid division by zero using expm1 and epsilon threshold
#             delta = z_U_up - z_L_up  # (P, BS, 3)
#             exp_z_L_up = torch.exp(z_L_up)
#             exp_z_U_up = torch.exp(z_U_up)
            
#             # Compute slope with safeguards
#             safe_slope = torch.zeros_like(delta)
#             mask = (torch.abs(delta) < eps)
#             # Case 1: delta ≈ 0 → use derivative exp(z_L_up)
#             safe_slope[mask] = exp_z_L_up[mask]
#             # Case 2: Use stable (exp(z_U_up) - exp(z_L_up)) / delta
#             safe_slope[~mask] = (exp_z_L_up[~mask] * torch.expm1(delta[~mask])) / delta[~mask]

#             # Upper bound coefficients
#             A_ub_prime = safe_slope.unsqueeze(-1) * A_ub  # (P, BS, 3, N)
#             b_ub_prime = safe_slope * (b_ub - z_L_up) + exp_z_L_up  # (P, BS, 3)

#             # ----------------------------------------------
#             # Step 4: Compute lower bound coefficients
#             # ----------------------------------------------
#             exp_z_L_low = torch.exp(z_L_low)
#             A_lb_prime = exp_z_L_low.unsqueeze(-1) * A_lb  # (P, BS, 3, N)
#             b_lb_prime = exp_z_L_low * (b_lb - z_L_low + 1)  # (P, BS, 3)

#             return A_lb_prime, A_ub_prime, b_lb_prime, b_ub_prime        
#         A_exp_lb, A_exp_ub, b_exp_lb, b_exp_ub = compute_exp_bounds(
#             A_prod_lb, A_prod_ub, b_prod_lb, b_prod_ub, alpha_lb[:,None,None], alpha_ub[:,None,None]
#         )          # P*BS*3*N, P*BS*3*N, P*BS*3, P*BS*3          
#         y_lb_min, _ = linear_bounds(A_exp_lb, b_exp_lb, alpha_lb, alpha_ub)
#         # y_lb_xl = (A_exp_lb * alpha_lb[:,None,None]).sum(dim=3) + b_exp_lb
#         # y_lb_xu = (A_exp_lb * alpha_ub[:,None,None]).sum(dim=3) + b_exp_lb
#         # y_lb_min = torch.where(A_exp_lb >= 0, y_lb_xl, y_lb_xu)  # (P, BS, N)                        
#         _, y_ub_min = linear_bounds(A_exp_ub, b_exp_ub, alpha_lb, alpha_ub)
#         # y_ub_xl = (A_exp_ub * alpha_lb[:,None,None]).sum(dim=3) + b_exp_ub
#         # y_ub_xu = (A_exp_ub * alpha_ub[:,None,None]).sum(dim=3) + b_exp_ub
#         # y_ub_max = torch.where(A_exp_ub >= 0, y_ub_xu, y_ub_xl)  # (P, BS, N)                        
#         # slope_ub=((torch.exp(prod_ub_ub)-torch.exp(prod_ub_lb))/(prod_ub_ub-prod_ub_lb))    # P*BS*3
#         # A_exp_ub = slope_ub.unsqueeze(-1)*A_prod_ub                                         # P*BS*3*N
#         # b_exp_ub = slope_ub*(b_prod_ub-prod_ub_lb)+torch.exp(prod_ub_lb)                    # P*BS*3
#         # slope_lb = torch.exp(prod_lb_lb)                                                    # P*BS*3
#         # A_exp_lb = slope_lb.unsqueeze(-1)*A_prod_lb                                         # P*BS*3*N
#         # b_exp_lb = slope_lb*(b_prod_lb-prod_lb_lb+1)                                        # P*BS*3

#         # Compute the final sum by adding all values along the BS dimension 
#         # The bound for final sum is given as (A_sum_lb*alpha).sum(dim=2)+b_sum_lb
#         A_sum_lb = A_exp_lb.sum(dim=1)              # P*3*N
#         A_sum_ub = A_exp_ub.sum(dim=1)              # P*3*N
#         b_sum_lb = b_exp_lb.sum(dim=1)              # P*3
#         b_sum_ub = b_exp_ub.sum(dim=1)              # P*3

#         A_result_lb = A_result_lb + A_sum_lb        # P*3*N
#         A_result_ub = A_result_ub + A_sum_ub        # P*3*N
#         b_result_lb = b_result_lb + b_sum_lb        # P*3
#         b_result_ub = b_result_ub + b_sum_ub        # P*3     

#     color_lb_lb, color_lb_ub = linear_bounds(A_result_lb[:,:,None], b_result_lb[:,:,None], alpha_lb, alpha_ub)
#     color_ub_lb, color_ub_ub = linear_bounds(A_result_ub[:,:,None], b_result_ub[:,:,None], alpha_lb, alpha_ub)
#     return color_lb_lb, color_ub_ub           

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

        assert torch.all(diffL<=diffU)

        step_L = torch.zeros(diffL.shape).to(A_ub.device)               # 1*N*BS
        mask_L = diffL>0
        step_L[mask_L] = 1.0
        step_U = torch.ones(diffU.shape).to(A_ub.device)                # 1*N*BS
        mask_U = diffU<=0
        step_U[mask_U] = 0.0
        # if not torch.all(step_L==step_U):
        #     print("stop here")

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
            tile_coord = tile_coord 
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
                    'iteration': 500, 
                    'lr_alpha':0.02, 
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
        tmp_lb_alpha, tmp_ub_alpha = model_alpha_bounded.compute_bounds(
            x=(inp_alpha, means_hom_tmp, cov_world, opacities_rast), 
            method='crown'
        )
        tmp_lb_alpha_emp = torch.zeros(tmp_lb_alpha.shape, device = tmp_lb_alpha.device)+1e10
        tmp_ub_alpha_emp = torch.zeros(tmp_ub_alpha.shape, device = tmp_ub_alpha.device)-1e10
        for i in range(1000):
            random_values = torch.rand_like(ptb_alpha.x_L)
            tmp_inp = torch.lerp(ptb_alpha.x_L, ptb_alpha.x_U, random_values)
            res = model_alpha(tmp_inp, means_hom_tmp, cov_world, opacities_rast) 
            tmp_lb_alpha_emp = torch.minimum(tmp_lb_alpha_emp, res)
            tmp_ub_alpha_emp = torch.maximum(tmp_ub_alpha_emp, res)
        tmp_lb_alpha2 = torch.minimum(tmp_lb_alpha, tmp_ub_alpha)
        tmp_ub_alpha2 = torch.maximum(tmp_lb_alpha, tmp_ub_alpha)
        # lb_alpha = tmp_lb_alpha
        # ub_alpha = tmp_ub_alpha
        lb_alpha = -tmp_ub_alpha2*0.5
        ub_alpha = -tmp_lb_alpha2*0.5
        lb_alpha = torch.exp(lb_alpha)
        ub_alpha = torch.exp(ub_alpha)
        ub_alpha = ub_alpha.clip(max=1.0)
        lb_alpha = lb_alpha*opacities_rast
        ub_alpha = ub_alpha*opacities_rast
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
        lb_depth, ub_depth, A_depth = model_depth_bounded.compute_bounds(x=(inp_depth, means_hom_tmp), method='crown', return_A=True, needed_A_dict=required_A)

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

    # step_L, step_U = get_bound_depth_step(
    #     ptb_depth,
    #     overall_depth_lA,
    #     overall_depth_uA,
    #     overall_depth_lbias,
    #     overall_depth_ubias,
    # )
    # tmp_res_Tl_old = computeT_new_optimized(overall_alpha_ub, step_U)
    # tmp_res_Tu_old = computeT_new_optimized(overall_alpha_lb, step_L)

    tmp_res_Tl, tmp_res_Tu = computeT_new_new(
        ptb_depth,
        overall_depth_lA,
        overall_depth_uA,
        overall_depth_lbias,
        overall_depth_ubias,   
        bounds_alpha[0:1],
        bounds_alpha[1:2]
    )


    tmp_color_Tl, tmp_color_Tu = computeT_new_new_new(
        ptb_depth,
        overall_depth_lA,
        overall_depth_uA,
        overall_depth_lbias,
        overall_depth_ubias,
        bounds_alpha[0:1],
        bounds_alpha[1:2], 
        colors_strip,
    )

    bounds_colors = torch.stack((colors_strip, colors_strip), dim=0)
    bounds_colors = bounds_colors[:,None]
    res_T = torch.cat((tmp_res_Tl, tmp_res_Tu), dim=0)
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

    eps_lb = torch.Tensor([[0,0,0,-0.0001,-0.0001,-0.0001]]).to(means.device)
    eps_ub = torch.Tensor([[0,0,0,0.0001,0.0001,0.0001]]).to(means.device)
    tile_size_global = 8
    gauss_step = 10000
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
    
    queue = [
        (h,w,min(h+initial_tilesize, height),min(w+initial_tilesize, width), initial_tilesize) \
        for w in range(0, width, initial_tilesize) for h in range(0, height, initial_tilesize)
    ]
    # Implement adaptive tile size 
    # while queue!=[]:
    #     hbl,wbl,htr,wtr,tile_size = queue[0]
    #     queue.pop(0)
    #     over_tl = rect[0][..., 0].clip(min=wbl), rect[0][..., 1].clip(min=hbl)
    #     over_br = rect[1][..., 0].clip(max=wtr-1), rect[1][..., 1].clip(max=htr-1)
    #     in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
    #     if not in_mask.sum() > 0:
    #         continue
    #     N = torch.where(in_mask)[0].shape[0]
    #     # If tile size too large or too much gaussians 
    #     if tile_size**2*N>threshold and tile_size>tile_size_global:
    #         if tile_size == 1:
    #             raise ValueError(f"Tile size can't be partitioned anymore, too many gaussians to be handled for ({hbl}, {wbl}), ({htr}, {wtr})")
    #         tile_size = tile_size//2 
    #         new_partitions = [
    #             (h,w,min(h+tile_size, htr),min(w+tile_size, wtr), tile_size) \
    #             for w in range(wbl, wtr, tile_size) for h in range(hbl, htr, tile_size)
    #         ]
    #         queue = queue+new_partitions 
    #         continue 
    #     means_strip = means[in_mask]
    #     quats_strip = quats[in_mask]
    #     opacities_strip = opacities[in_mask]
    #     scales_strip = scales[in_mask]
    #     colors_strip = colors[in_mask]
    #     print(f">>>>>>>> {hbl}, {wbl}, {htr}, {wtr}, {means_strip.shape[0]}")

    #     tile_color_lb, tile_color_ub = compute_tile_color(
    #         cam_inp,
    #         eps_lb,
    #         eps_ub,
    #         means_strip,
    #         opacities_strip,
    #         scales_strip,
    #         quats_strip,
    #         colors_strip,
    #         f,
    #         wbl,
    #         hbl,
    #         wtr,
    #         htr,
    #         width,
    #         height,
    #         pix_coord,
    #         tile_size,
    #         gauss_step
    #     )
    #     render_color_lb[hbl:htr, wbl:wtr] = tile_color_lb
    #     render_color_ub[hbl:htr, wbl:wtr] = tile_color_ub
    #     plt.imshow(render_color_lb)
    #     plt.savefig('res_lb.png')
    #     plt.imshow(render_color_ub)
    #     plt.savefig('res_ub.png')
        

    for h in range(0, height, tile_size_global):
        for w in range(0, width, tile_size_global):
            # if h!=0 or w!=38:
            #     continue
            # if h>24:
            #     continue
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+tile_size_global-1), rect[1][..., 1].clip(max=h+tile_size_global-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
            if not in_mask.sum() > 0:
                continue
            means_strip = means[in_mask]
            quats_strip = quats[in_mask]
            opacities_strip = opacities[in_mask]
            scales_strip = scales[in_mask]
            colors_strip = colors[in_mask]
            print(f">>>>>>>> {h}, {w}, {means_strip.shape[0]}")

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
                w,
                h,
                w+tile_size_global,
                h+tile_size_global,
                width,
                height,
                pix_coord,
                tile_size_global,
                gauss_step
            )
            render_color_lb[h:h+tile_size_global, w:w+tile_size_global] = tile_color_lb
            render_color_ub[h:h+tile_size_global, w:w+tile_size_global] = tile_color_ub
            plt.imshow(render_color_lb)
            plt.savefig('res_lb.png')
            plt.imshow(render_color_ub)
            plt.savefig('res_ub.png')
    plt.figure(1)
    plt.imshow(render_color_lb)
    plt.figure(2)
    plt.imshow(render_color_ub)
    plt.show()
