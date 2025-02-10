import torch
# import math
# from gaussian_splatting.utils.sh_utils import eval_sh

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    # assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    # assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] + # 
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def build_color(means3D, shs, c2w, active_sh_degree):
    camtoworlds = torch.linalg.inv(c2w)
    rays_o = camtoworlds[:3,3]
    rays_d = means3D - rays_o
    rays_d = rays_d/rays_d.norm(dim=1, keepdim=True)
    # shs_norms = torch.norm(shs, dim=1, keepdim = True)
    # shs_norms = torch.clamp(shs_norms, min=1e-8)
    # shs_normalized = shs/shs_norms
    # color = eval_sh(active_sh_degree, shs.permute(0,2,1), rays_d)
    sh_coeffs = eval_sh_old(active_sh_degree, rays_d)
    sh_coeffs_expanded = sh_coeffs.unsqueeze(2)  # Shape: (424029, 16, 1)
    product = sh_coeffs_expanded * shs
    color = torch.sum(product, dim=1)
    color = (color + 0.5).clip(min=0.0)
    return color

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

def render_notile(means2D, cov2D, radii, color, opacity, depths, W, H, device='cuda', tile_size = 64):
    pix_coord = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device)
    # radii = get_radius(cov2D)    
    render_color = torch.ones(*pix_coord.shape[:2], 3).to(device)
    render_depth = torch.zeros(*pix_coord.shape[:2], 1).to(device)
    render_alpha = torch.zeros(*pix_coord.shape[:2], 1).to(device)

    # tile_size = 64
    # for h in range(0, H, tile_size):
    #     for w in range(0, W, tile_size):
    # check if the rectangle penetrate the tile
    # over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
    # over_br = rect[1][..., 0].clip(max=w+tile_size-1), rect[1][..., 1].clip(max=h+tile_size-1)
    # in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
    # in_mask = torch.full((rect[0].shape[0],), True, dtype=torch.bool)

    # P = in_mask.sum()
    tile_coord = pix_coord.flatten(0,-2)
    sorted_depths, index = torch.sort(depths)
    sorted_means2D = means2D[index]
    sorted_cov2D = cov2D[index] # P 2 2
    sorted_conic = sorted_cov2D.inverse() # inverse of variance
    sorted_opacity = opacity[index].unsqueeze(1)
    sorted_color = color[index]
    dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
    dx_unsorted = (tile_coord[:,None,:] - means2D[None,:])
    conic = cov2D.inverse()

    gauss_weight = torch.exp(-0.5 * (
        dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
        + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
        + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
        + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
    gauss_weight_unsorted = torch.exp(-0.5 * (
        dx_unsorted[:, :, 0]**2 * conic[:, 0, 0] 
        + dx_unsorted[:, :, 1]**2 * conic[:, 1, 1]
        + dx_unsorted[:,:,0]*dx_unsorted[:,:,1] * conic[:, 0, 1]
        + dx_unsorted[:,:,0]*dx_unsorted[:,:,1] * conic[:, 1, 0]))

    alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
    alpha_unsorted = (gauss_weight_unsorted[..., None] * opacity[None,:,None]).clip(max=0.99)
    T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
    acc_alpha = (alpha * T).sum(dim=1)
    tile_color = (T * alpha * sorted_color[None]).sum(dim=1) # + (1-acc_alpha) * (1 if white_bkgd else 0)
    tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
    # tile_h, tile_w = render_color[h:h+tile_size, w:w+tile_size].shape[0],render_color[h:h+tile_size, w:w+tile_size].shape[1]
    render_color = tile_color.reshape(H, W, -1)
    render_depth = tile_depth.reshape(H, W, -1)
    render_alpha = acc_alpha.reshape(H, W, -1)

    return {
        "render": render_color,
        "depth": render_depth,
        "alpha": render_alpha,
        "visiility_filter": radii > 0,
        "radii": radii
    }


def render(means2D, cov2D, radii, color, opacity, depths, W, H, device='cuda', tile_size = 64):
    pix_coord = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device)
    # radii = get_radius(cov2D)
    rect = get_rect(means2D, radii, width=W, height=H)
    
    render_color = torch.zeros(*pix_coord.shape[:2], color.shape[-1]).to(device)
    render_depth = torch.zeros(*pix_coord.shape[:2], 1).to(device)
    render_alpha = torch.zeros(*pix_coord.shape[:2], 1).to(device)

    # tile_size = 64
    for h in range(0, H, tile_size):
        for w in range(0, W, tile_size):
            # check if the rectangle penetrate the tile
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+tile_size-1), rect[1][..., 1].clip(max=h+tile_size-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
            # in_mask = torch.full((rect[0].shape[0],), True, dtype=torch.bool)

            if not in_mask.sum() > 0:
                continue

            P = in_mask.sum()
            tile_coord = pix_coord[h:h+tile_size, w:w+tile_size].flatten(0,-2)
            sorted_depths, index = torch.sort(depths[in_mask])
            sorted_means2D = means2D[in_mask][index]
            sorted_cov2D = cov2D[in_mask][index] # P 2 2
            sorted_conic = sorted_cov2D.inverse() # inverse of variance
            sorted_opacity = opacity[in_mask][index].unsqueeze(1)
            sorted_color = color[in_mask][index]
            dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
            
            gauss_weight = torch.exp(-0.5 * (
                dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
            
            alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
            T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
            acc_alpha = (alpha * T).sum(dim=1)
            tile_color = (T * alpha * sorted_color[None]).sum(dim=1) # + (1-acc_alpha) * (1 if white_bkgd else 0)
            tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
            tile_h, tile_w = render_color[h:h+tile_size, w:w+tile_size].shape[0],render_color[h:h+tile_size, w:w+tile_size].shape[1]
            render_color[h:h+tile_h, w:w+tile_w] = tile_color.reshape(tile_h, tile_w, -1)
            render_depth[h:h+tile_h, w:w+tile_w] = tile_depth.reshape(tile_h, tile_w, -1)
            render_alpha[h:h+tile_h, w:w+tile_w] = acc_alpha.reshape(tile_h, tile_w, -1)

    return {
        "render": render_color,
        "depth": render_depth,
        "alpha": render_alpha,
        "visiility_filter": radii > 0,
        "radii": radii
    }

def render_mask(means2D, cov2D, radii, color, opacity, depths, W, H, device='cuda', tile_size = 64):
    pix_coord = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device)
    # radii = get_radius(cov2D)
    rect = get_rect(means2D, radii, width=W, height=H)
    
    render_color = torch.ones(*pix_coord.shape[:2], 3).to(device)
    render_depth = torch.zeros(*pix_coord.shape[:2], 1).to(device)
    render_alpha = torch.zeros(*pix_coord.shape[:2], 1).to(device)

    min_inmask_len = float('inf')
    # tile_size = 64
    min_inmask = None
    for h in range(0, H, tile_size):
        for w in range(0, W, tile_size):
            # check if the rectangle penetrate the tile
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+tile_size-1), rect[1][..., 1].clip(max=h+tile_size-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
            # in_mask = torch.full((rect[0].shape[0],), True, dtype=torch.bool)

            if not in_mask.sum() > 0:
                continue

            P = in_mask.sum()
            tile_coord = pix_coord[h:h+tile_size, w:w+tile_size].flatten(0,-2)
            sorted_depths, index = torch.sort(depths[in_mask])
            sorted_means2D = means2D[in_mask][index]
            sorted_cov2D = cov2D[in_mask][index] # P 2 2
            sorted_conic = sorted_cov2D.inverse() # inverse of variance
            sorted_opacity = opacity[in_mask][index].unsqueeze(1)
            sorted_color = color[in_mask][index]
            dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
            
            if len(torch.where(in_mask)[0])<min_inmask_len:
                min_inmask_len = len(torch.where(in_mask)[0])
                min_inmask = in_mask
            
            # gauss_weight = torch.exp(-0.5 * (
            #     dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
            #     + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
            #     + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
            #     + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
            
            # alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
            # T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
            # acc_alpha = (alpha * T).sum(dim=1)
            # tile_color = (T * alpha * sorted_color[None]).sum(dim=1) # + (1-acc_alpha) * (1 if white_bkgd else 0)
            # tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
            # tile_h, tile_w = render_color[h:h+tile_size, w:w+tile_size].shape[0],render_color[h:h+tile_size, w:w+tile_size].shape[1]
            # render_color[h:h+tile_h, w:w+tile_w] = tile_color.reshape(tile_h, tile_w, -1)
            # render_depth[h:h+tile_h, w:w+tile_w] = tile_depth.reshape(tile_h, tile_w, -1)
            # render_alpha[h:h+tile_h, w:w+tile_w] = acc_alpha.reshape(tile_h, tile_w, -1)
    return min_inmask
    # return {
    #     "render": render_color,
    #     "depth": render_depth,
    #     "alpha": render_alpha,
    #     "visiility_filter": radii > 0,
    #     "radii": radii
    # }

def quaternion_to_rotation_matrix(quats):
    """
    Converts quaternions to rotation matrices.
    Args:
        quats: Tensor of shape [N, 4], where each quaternion is [w, x, y, z].
    Returns:
        Rotation matrices of shape [N, 3, 3].
    """
    quats = quats / quats.norm(dim=1, keepdim=True)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    N = quats.size(0)
    R = torch.empty(N, 3, 3, device=quats.device, dtype=quats.dtype)

    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)

    return R

def eval_sh_old(degree, dirs):
    """
    Evaluate real spherical harmonics (SH) up to given degree at directions.
    Args:
        degree: int, degree of SH (e.g., 3)
        dirs: Tensor of shape [..., 3], unit vectors
    Returns:
        SH basis functions evaluated at dirs, shape [..., (degree+1)**2]
    """
    # assert degree >= 0 and degree <= 3, "Only degrees 0 to 3 are supported"
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    result = []
    if degree >= 0:
        result.append(0.282095 * torch.ones_like(x))  # Y_0^0

    if degree >= 1:
        result.append(-0.488603 * y)  # Y_1^-1
        result.append(0.488603 * z)   # Y_1^0
        result.append(-0.488603 * x)  # Y_1^1

    if degree >= 2:
        result.append(1.092548 * x * y)                     # Y_2^-2
        result.append(-1.092548 * y * z)                    # Y_2^-1
        result.append(0.315392 * (3 * z**2 - 1))            # Y_2^0
        result.append(-1.092548 * x * z)                    # Y_2^1
        result.append(0.546274 * (x**2 - y**2))             # Y_2^2

    if degree >= 3:
        result.append(-0.590044 * y * (3 * x**2 - y**2))            # Y_3^-3
        result.append(2.890611 * x * y * z)                         # Y_3^-2
        result.append(-0.457046 * y * (5 * z**2 - 1))               # Y_3^-1
        result.append(0.373176 * z * (5 * z**2 - 3))                # Y_3^0
        result.append(-0.457046 * x * (5 * z**2 - 1))               # Y_3^1
        result.append(1.445306 * z * (x**2 - y**2))                 # Y_3^2
        result.append(-0.590044 * x * (x**2 - 3 * y**2))            # Y_3^3

    sh_basis = torch.stack(result, dim=-1)  # [..., K]
    return sh_basis

def rasterize_gaussians_pytorch(
        means, 
        quats, 
        scales, 
        opacities, 
        colors, 
        viewmats, 
        Ks, 
        width, 
        height, 
        near_plane=0.01, 
        far_plane=1e10, 
        sh_degree=3, 
        tile_size=16,
        eps2d = 0.3,
        radius_clip = 0.0
    ):
    device = means.device
    dtype = means.dtype

    # For simplicity, we'll render from the first camera
    viewmat = viewmats[0]  # [4, 4]
    K = Ks[0]              # [3, 3]

    N = means.size(0)

    # Step 1: Transform Gaussian centers to camera space
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    means_hom = torch.cat([means, ones], dim=1).to(device)  # [N, 4]
    means_cam_hom = (viewmat @ means_hom.T).T    # [N, 4]
    means_cam = means_cam_hom[:, :3] / means_cam_hom[:, 3:4]  # [N, 3]

    mask = (means_cam[:,2]>near_plane) & (means_cam[:,2]<far_plane)
    overall_mask = mask
    # means = means[mask]
    # means_cam = means_cam[mask]
    # quats = quats[mask] 
    # scales = scales[mask] 
    # opacities = opacities[mask] 
    # colors = colors[mask] 
    # N = means_cam.size(0)  # Update N after masking

    # Step 2: Compute rotation matrices from quaternions
    R_gaussians = quaternion_to_rotation_matrix(quats)  # [N, 3, 3]

    # Step 3: Compute covariance matrices in world space
    scales_matrix = torch.diag_embed(scales).to(device)  # [N, 3, 3]
    M = R_gaussians@scales_matrix
    cov_world = M @ M.transpose(1, 2)  # [N, 3, 3]

    # Step 4: Transform covariance matrices to camera space
    R_cam = viewmat[:3, :3]  # [3, 3]
    R_cam_expanded = R_cam.unsqueeze(0).expand(N, 3, 3)
    cov_cam = R_cam_expanded @ cov_world @ R_cam_expanded.transpose(1, 2)  # [N, 3, 3]

    # Step 5: Project means onto the image plane
    means_proj_hom = (K @ means_cam.T).T  # [N, 3]
    means2D = means_proj_hom[:, :2] / means_proj_hom[:, 2:3]  # [N, 2]

    # Step 6: Compute 2D covariance matrices using the Jacobian
    fx = K[0, 0]
    fy = K[1, 1]
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z_cam = means_cam[:, 2]

    tan_fovx = 0.5*width/fx 
    tan_fovy = 0.5*height/fy 
    lim_x = 1.3*tan_fovx 
    lim_y = 1.3*tan_fovy

    tx = z_cam*torch.min(lim_x, torch.max(-lim_x, x/z_cam))
    ty = z_cam*torch.min(lim_y, torch.max(-lim_y, y/z_cam))

    J = torch.zeros(N, 2, 3, device=device, dtype=dtype)
    J[:, 0, 0] = fx / z_cam
    J[:, 0, 2] = -fx * tx / z_cam**2
    J[:, 1, 1] = fy / z_cam
    J[:, 1, 2] = -fy * ty / z_cam**2

    cov2D = J @ cov_cam @ J.transpose(1, 2)  # [N, 2, 2]

    det_orig = cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
    cov2D[:,0,0] = cov2D[:,0,0]+eps2d 
    cov2D[:,1,1] = cov2D[:,1,1]+eps2d 
    det_blur = cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
    compensation = torch.sqrt(torch.max(torch.zeros((det_orig/det_blur).shape).to(det_orig.device), det_orig/det_blur))
    det = det_blur

    mask = det_blur>0
    overall_mask = overall_mask & mask 
    # means = means[mask]
    # means_cam = means_cam[mask]
    # cov2D = cov2D[mask]
    # means2D = means2D[mask]
    # opacities = opacities[mask]
    # colors = colors[mask]
    # N = means2D.size(0)  # Update N after masking

    cov2D_inv = torch.linalg.inv(cov2D)

    # Step 7: Check if points are in image region 
    # Take 3 sigma as the radius 
    b = 0.5*(cov2D[:,0,0]+cov2D[:,1,1])
    v1 = b+torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
    v2 = b-torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
    radius = torch.ceil(3*torch.sqrt(torch.max(v1, v2)))
    
    mask = radius>radius_clip
    overall_mask = overall_mask & mask 
    # means = means[mask]
    # means_cam = means_cam[mask]
    # cov2D = cov2D[mask]
    # means2D = means2D[mask]
    # opacities = opacities[mask]
    # colors = colors[mask]
    # radius = radius[mask]
    # N = means2D.size(0)  # Update N after masking

    # mask out gaussians outside the image region
    mask = (means2D[:,0]+radius>0) & (means2D[:,0]-radius<width) & (means2D[:,1]+radius>0) & (means2D[:,1]-radius<height)
    overall_mask = overall_mask & mask 
    means = means[overall_mask]
    means_cam = means_cam[overall_mask]
    cov2D = cov2D[overall_mask]
    means2D = means2D[overall_mask]
    opacities = opacities[overall_mask]
    colors = colors[overall_mask]
    radius = radius[overall_mask]
    N = means2D.size(0)  # Update N after masking

    color_rgb = build_color(means3D=means, shs=colors, c2w=viewmat, active_sh_degree=3)
    # return color_rgb

    # with torch.no_grad():
    res = render(
        means2D=means2D,
        cov2D=cov2D,
        radii=radius,
        color=color_rgb,
        opacity=opacities,
        depths = means_cam[:,2],
        W=width,
        H=height,
        tile_size=tile_size
    )
    return res['render']

def rasterize_gaussians_pytorch_rgb(
        means, 
        quats, 
        scales, 
        opacities, 
        colors, 
        viewmats, 
        Ks, 
        width, 
        height, 
        near_plane=0.01, 
        far_plane=1e10, 
        sh_degree=3, 
        tile_size=8,
        eps2d = 0.3,
        radius_clip = 0.0
    ):
    device = means.device
    dtype = means.dtype

    # For simplicity, we'll render from the first camera
    viewmat = viewmats[0]  # [4, 4]
    K = Ks[0]              # [3, 3]

    N = means.size(0)

    # Step 1: Transform Gaussian centers to camera space
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    means_hom = torch.cat([means, ones], dim=1).to(device)  # [N, 4]
    means_cam_hom = (viewmat @ means_hom.T).T    # [N, 4]
    means_cam = means_cam_hom[:, :3] / means_cam_hom[:, 3:4]  # [N, 3]

    mask = (means_cam[:,2]>near_plane) & (means_cam[:,2]<far_plane)
    overall_mask = mask
    # means = means[mask]
    # means_cam = means_cam[mask]
    # quats = quats[mask] 
    # scales = scales[mask] 
    # opacities = opacities[mask] 
    # colors = colors[mask] 
    # N = means_cam.size(0)  # Update N after masking

    # Step 2: Compute rotation matrices from quaternions
    R_gaussians = quaternion_to_rotation_matrix(quats)  # [N, 3, 3]

    # Step 3: Compute covariance matrices in world space
    scales_matrix = torch.diag_embed(scales).to(device)  # [N, 3, 3]
    M = R_gaussians@scales_matrix
    cov_world = M @ M.transpose(1, 2)  # [N, 3, 3]

    # Step 4: Transform covariance matrices to camera space
    R_cam = viewmat[:3, :3]  # [3, 3]
    R_cam_expanded = R_cam.unsqueeze(0).expand(N, 3, 3)
    cov_cam = R_cam_expanded @ cov_world @ R_cam_expanded.transpose(1, 2)  # [N, 3, 3]

    # Step 5: Project means onto the image plane
    means_proj_hom = (K @ means_cam.T).T  # [N, 3]
    means2D = means_proj_hom[:, :2] / means_proj_hom[:, 2:3]  # [N, 2]

    # Step 6: Compute 2D covariance matrices using the Jacobian
    fx = K[0, 0]
    fy = K[1, 1]
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z_cam = means_cam[:, 2]

    tan_fovx = 0.5*width/fx 
    tan_fovy = 0.5*height/fy 
    lim_x = 1.3*tan_fovx 
    lim_y = 1.3*tan_fovy

    tx = z_cam*torch.min(lim_x, torch.max(-lim_x, x/z_cam))
    ty = z_cam*torch.min(lim_y, torch.max(-lim_y, y/z_cam))

    J = torch.zeros(N, 2, 3, device=device, dtype=dtype)
    J[:, 0, 0] = fx / z_cam
    J[:, 0, 2] = -fx * tx / z_cam**2
    J[:, 1, 1] = fy / z_cam
    J[:, 1, 2] = -fy * ty / z_cam**2

    cov2D = J @ cov_cam @ J.transpose(1, 2)  # [N, 2, 2]

    det_orig = cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
    cov2D[:,0,0] = cov2D[:,0,0]+eps2d 
    cov2D[:,1,1] = cov2D[:,1,1]+eps2d 
    det_blur = cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
    compensation = torch.sqrt(torch.max(torch.zeros((det_orig/det_blur).shape).to(det_orig.device), det_orig/det_blur))
    # opacities = opacities * compensation
    det = det_blur

    mask = det_blur>0
    overall_mask = overall_mask & mask 
    # means = means[mask]
    # means_cam = means_cam[mask]
    # cov2D = cov2D[mask]
    # means2D = means2D[mask]
    # opacities = opacities[mask]
    # colors = colors[mask]
    # N = means2D.size(0)  # Update N after masking

    cov2D_inv = torch.linalg.inv(cov2D)

    # Step 7: Check if points are in image region 
    # Take 3 sigma as the radius 
    b = 0.5*(cov2D[:,0,0]+cov2D[:,1,1])
    v1 = b+torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
    v2 = b-torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
    radius = torch.ceil(3*torch.sqrt(torch.max(v1, v2)))
    
    mask = radius>radius_clip
    overall_mask = overall_mask & mask 
    # means = means[mask]
    # means_cam = means_cam[mask]
    # cov2D = cov2D[mask]
    # means2D = means2D[mask]
    # opacities = opacities[mask]
    # colors = colors[mask]
    # radius = radius[mask]
    # N = means2D.size(0)  # Update N after masking

    # mask out gaussians outside the image region
    # mask = (means2D[:,0]+radius>0) & (means2D[:,0]-radius<width) & (means2D[:,1]+radius>0) & (means2D[:,1]-radius<height)
    overall_mask = overall_mask & mask 
    means = means[overall_mask]
    means_cam = means_cam[overall_mask]
    cov2D = cov2D[overall_mask]
    means2D = means2D[overall_mask]
    opacities = opacities[overall_mask]
    colors = colors[overall_mask]
    radius = radius[overall_mask]
    N = means2D.size(0)  # Update N after masking

    # color_rgb = build_color(means3D=means, shs=colors, c2w=viewmat, active_sh_degree=3)
    color_rgb = colors
    # return color_rgb

    # with torch.no_grad():
    res = render(
        means2D=means2D,
        cov2D=cov2D,
        radii=radius,
        color=color_rgb,
        opacity=opacities,
        depths = means_cam[:,2],
        W=width,
        H=height,
        tile_size=tile_size
    )
    res['overall_mask'] = overall_mask
    return res

def rasterize_gaussians_debug(
        means, 
        quats, 
        scales, 
        opacities, 
        colors, 
        viewmats, 
        Ks, 
        width, 
        height, 
        near_plane=0.01, 
        far_plane=1e10, 
        sh_degree=3, 
        tile_size=16,
        eps2d = 0.3,
        radius_clip = 0.0
    ):
    device = means.device
    dtype = means.dtype

    # For simplicity, we'll render from the first camera
    viewmat = viewmats[0]  # [4, 4]
    K = Ks[0]              # [3, 3]

    N = means.size(0)

    # Step 1: Transform Gaussian centers to camera space
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    means_hom = torch.cat([means, ones], dim=1).to(device)  # [N, 4]
    means_cam_hom = (viewmat @ means_hom.T).T    # [N, 4]
    means_cam = means_cam_hom[:, :3] / means_cam_hom[:, 3:4]  # [N, 3]

    mask = (means_cam[:,2]>near_plane) & (means_cam[:,2]<far_plane)
    overall_mask = mask
    # means = means[mask]
    # means_cam = means_cam[mask]
    # quats = quats[mask] 
    # scales = scales[mask] 
    # opacities = opacities[mask] 
    # colors = colors[mask] 
    # N = means_cam.size(0)  # Update N after masking

    # Step 2: Compute rotation matrices from quaternions
    R_gaussians = quaternion_to_rotation_matrix(quats)  # [N, 3, 3]

    # Step 3: Compute covariance matrices in world space
    scales_matrix = torch.diag_embed(scales).to(device)  # [N, 3, 3]
    M = R_gaussians@scales_matrix
    cov_world = M @ M.transpose(1, 2)  # [N, 3, 3]

    # Step 4: Transform covariance matrices to camera space
    R_cam = viewmat[:3, :3]  # [3, 3]
    R_cam_expanded = R_cam.unsqueeze(0).expand(N, 3, 3)
    cov_cam = R_cam_expanded @ cov_world @ R_cam_expanded.transpose(1, 2)  # [N, 3, 3]

    # Step 5: Project means onto the image plane
    means_proj_hom = (K @ means_cam.T).T  # [N, 3]
    means2D = means_proj_hom[:, :2] / means_proj_hom[:, 2:3]  # [N, 2]

    # Step 6: Compute 2D covariance matrices using the Jacobian
    fx = K[0, 0]
    fy = K[1, 1]
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z_cam = means_cam[:, 2]

    tan_fovx = 0.5*width/fx 
    tan_fovy = 0.5*height/fy 
    lim_x = 1.3*tan_fovx 
    lim_y = 1.3*tan_fovy

    tx = z_cam*torch.min(lim_x, torch.max(-lim_x, x/z_cam))
    ty = z_cam*torch.min(lim_y, torch.max(-lim_y, y/z_cam))

    J = torch.zeros(N, 2, 3, device=device, dtype=dtype)
    J[:, 0, 0] = fx / z_cam
    J[:, 0, 2] = -fx * tx / z_cam**2
    J[:, 1, 1] = fy / z_cam
    J[:, 1, 2] = -fy * ty / z_cam**2

    cov2D = J @ cov_cam @ J.transpose(1, 2)  # [N, 2, 2]

    det_orig = cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
    cov2D[:,0,0] = cov2D[:,0,0]+eps2d 
    cov2D[:,1,1] = cov2D[:,1,1]+eps2d 
    det_blur = cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
    compensation = torch.sqrt(torch.max(torch.zeros((det_orig/det_blur).shape).to(det_orig.device), det_orig/det_blur))
    det = det_blur

    mask = det_blur>0
    overall_mask = overall_mask & mask 
    # means = means[mask]
    # means_cam = means_cam[mask]
    # cov2D = cov2D[mask]
    # means2D = means2D[mask]
    # opacities = opacities[mask]
    # colors = colors[mask]
    # N = means2D.size(0)  # Update N after masking

    cov2D_inv = torch.linalg.inv(cov2D)

    # Step 7: Check if points are in image region 
    # Take 3 sigma as the radius 
    b = 0.5*(cov2D[:,0,0]+cov2D[:,1,1])
    v1 = b+torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
    v2 = b-torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
    radius = torch.ceil(3*torch.sqrt(torch.max(v1, v2)))
    
    mask = radius>radius_clip
    overall_mask = overall_mask & mask 
    # means = means[mask]
    # means_cam = means_cam[mask]
    # cov2D = cov2D[mask]
    # means2D = means2D[mask]
    # opacities = opacities[mask]
    # colors = colors[mask]
    # radius = radius[mask]
    # N = means2D.size(0)  # Update N after masking

    # mask out gaussians outside the image region
    mask = (means2D[:,0]+radius>0) & (means2D[:,0]-radius<width) & (means2D[:,1]+radius>0) & (means2D[:,1]-radius<height)
    overall_mask = overall_mask & mask 
    means = means[overall_mask]
    means_cam = means_cam[overall_mask]
    cov2D = cov2D[overall_mask]
    means2D = means2D[overall_mask]
    opacities = opacities[overall_mask]
    colors = colors[overall_mask]
    radius = radius[overall_mask]
    N = means2D.size(0)  # Update N after masking

    color_rgb = build_color(means3D=means, shs=colors, c2w=viewmat, active_sh_degree=3)
    # return color_rgb

    # with torch.no_grad():
    res = render_mask(
        means2D=means2D,
        cov2D=cov2D,
        radii=radius,
        color=color_rgb,
        opacity=opacities,
        depths = means_cam[:,2],
        W=width,
        H=height,
        tile_size=tile_size
    )
    return res, means, means2D, viewmat


    #==========================================================================================

    # # Compute radii for bounding boxes
    # eigenvalues = torch.linalg.eigvalsh(cov2D)  # [N, 2]
    # radii = 2 * torch.sqrt(eigenvalues.max(dim=1).values)  # [N]

    # # Compute bounding rectangles
    # x_min = (means2D[:, 0] - radii).clamp(0, width - 1)
    # x_max = (means2D[:, 0] + radii).clamp(0, width - 1)
    # y_min = (means2D[:, 1] - radii).clamp(0, height - 1)
    # y_max = (means2D[:, 1] + radii).clamp(0, height - 1)

    # rects = ((x_min, y_min), (x_max, y_max))  # Each is [N]

    # # Prepare render buffers
    # render_color = torch.zeros(height, width, 3, device=device, dtype=dtype)
    # render_depth = torch.zeros(height, width, 1, device=device, dtype=dtype)
    # render_alpha = torch.zeros(height, width, 1, device=device, dtype=dtype)

    # # Prepare pixel grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    # pix_coord = torch.stack([grid_x, grid_y], dim=-1).float()  # [height, width, 2]

    # # Rasterization loop
    # for h in range(0, height, tile_size):
    #     for w in range(0, width, tile_size):
    #         tile_pix_coord = pix_coord[h:h+tile_size, w:w+tile_size].reshape(-1, 2)  # [B, 2]

    #         # Check which Gaussians overlap with the tile
    #         over_tl_x = rects[0][0].clamp(min=w)
    #         over_tl_y = rects[0][1].clamp(min=h)
    #         over_br_x = rects[1][0].clamp(max=w+tile_size-1)
    #         over_br_y = rects[1][1].clamp(max=h+tile_size-1)

    #         in_mask = (over_br_x > over_tl_x) & (over_br_y > over_tl_y)

    #         if not in_mask.any():
    #             continue

    #         P = in_mask.sum()

    #         # Get the Gaussians in the tile
    #         means2D_in = means2D[in_mask]  # [P, 2]
    #         cov2D_in = cov2D[in_mask]      # [P, 2, 2]
    #         depths_in = z[in_mask]         # [P]
    #         opacities_in = opacities[in_mask]  # [P]
    #         colors_in = colors[in_mask]        # [P, K, 3]
    #         means_cam_in = means_cam[in_mask]  # [P, 3]

    #         # Compute the inverse of covariance matrices
    #         cov2D_inv = torch.inverse(cov2D_in)  # [P, 2, 2]

    #         # Compute distances between pixels and Gaussian centers
    #         dx = tile_pix_coord[:, None, :] - means2D_in[None, :, :]  # [B, P, 2]

    #         # Compute Gaussian weights
    #         exponents = torch.einsum('bpi,pij,bpj->bp', dx, cov2D_inv, dx)  # [B, P]
    #         gauss_weight = torch.exp(-0.5 * exponents)  # [B, P]

    #         # Compute alpha values
    #         alpha = (gauss_weight * opacities_in[None, :]).clamp(max=0.99)  # [B, P]

    #         # Compute directions from Gaussians to pixels (in camera space)
    #         # For each pixel and Gaussian, compute direction vector
    #         pixel_coords_h = torch.cat([tile_pix_coord, torch.ones_like(tile_pix_coord[:, :1])], dim=1)  # [B, 3]
    #         pixel_coords_cam = torch.inverse(K) @ pixel_coords_h.T  # [3, B]
    #         pixel_coords_cam = pixel_coords_cam.T  # [B, 3]
    #         pixel_dirs = pixel_coords_cam / pixel_coords_cam.norm(dim=1, keepdim=True)  # [B, 3]

    #         # Expand pixel directions and Gaussian positions
    #         pixel_dirs_exp = pixel_dirs[:, None, :].expand(-1, P, -1)  # [B, P, 3]
    #         means_cam_in_exp = means_cam_in[None, :, :].expand(tile_pix_coord.size(0), -1, -1)  # [B, P, 3]

    #         # Compute directions from Gaussians to pixels (unit vectors)
    #         dirs = pixel_dirs_exp - means_cam_in_exp  # [B, P, 3]
    #         dirs = dirs / dirs.norm(dim=-1, keepdim=True)  # [B, P, 3]

    #         # Evaluate SH basis functions at these directions
    #         dirs_flat = dirs.reshape(-1, 3)  # [B*P, 3]
    #         sh_basis = eval_sh(sh_degree, dirs_flat)  # [B*P, K]
    #         K_sh = sh_basis.size(-1)
    #         sh_basis = sh_basis.reshape(tile_pix_coord.size(0), P, K_sh)  # [B, P, K]

    #         # Compute colors using SH coefficients
    #         # colors_in: [P, K, 3]
    #         colors_in = colors_in.unsqueeze(0).expand(tile_pix_coord.size(0), -1, -1, -1)  # [B, P, K, 3]
    #         sh_basis = sh_basis.unsqueeze(-1)  # [B, P, K, 1]
    #         color_contrib = (colors_in * sh_basis).sum(dim=2)  # [B, P, 3]

    #         # Sort Gaussians by depth (front to back)
    #         sorted_depths, indices = depths_in.sort()
    #         gauss_weight = gauss_weight[:, indices]
    #         alpha = alpha[:, indices]
    #         color_contrib = color_contrib[:, indices, :]  # [B, P, 3]

    #         # Compute transmittance T and accumulated alpha
    #         T = torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1).cumprod(dim=1)  # [B, P]
    #         acc_alpha = (alpha * T).sum(dim=1, keepdim=True)  # [B, 1]

    #         # Compute accumulated color
    #         tile_color = (T[:, :, None] * alpha[:, :, None] * color_contrib).sum(dim=1)  # [B, 3]

    #         # Compute accumulated depth
    #         tile_depth = (T * alpha * sorted_depths[None, :]).sum(dim=1, keepdim=True)  # [B, 1]

    #         # Reshape to tile_size x tile_size
    #         B = tile_pix_coord.size(0)
    #         size_y = min(tile_size, height - h)
    #         size_x = min(tile_size, width - w)
    #         tile_color = tile_color.reshape(size_y, size_x, -1)
    #         tile_depth = tile_depth.reshape(size_y, size_x, -1)
    #         acc_alpha = acc_alpha.reshape(size_y, size_x, -1)

    #         # Update render buffers
    #         render_color[h:h+size_y, w:w+size_x] = tile_color
    #         render_depth[h:h+size_y, w:w+size_x] = tile_depth
    #         render_alpha[h:h+size_y, w:w+size_x] = acc_alpha

    # return {
    #     "render": render_color,
    #     "depth": render_depth,
    #     "alpha": render_alpha
    # }

if __name__ == "__main__":
    pass 