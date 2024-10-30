import torch

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

def rasterize_gaussians(means, quats, scales, opacities, colors, viewmats, Ks, width, height, near_plane=0.01, far_plane=1e10, sh_degree=3, tile_size=16):
    device = means.device
    dtype = means.dtype

    # For simplicity, we'll render from the first camera
    viewmat = viewmats[0]  # [4, 4]
    K = Ks[0]              # [3, 3]

    N = means.size(0)

    # Step 1: Transform Gaussian centers to camera space
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    means_hom = torch.cat([means, ones], dim=1)  # [N, 4]
    means_cam_hom = (viewmat @ means_hom.T).T    # [N, 4]
    means_cam = means_cam_hom[:, :3] / means_cam_hom[:, 3:4]  # [N, 3]

    # Step 2: Compute rotation matrices from quaternions
    R_gaussians = quaternion_to_rotation_matrix(quats)  # [N, 3, 3]

    # Step 3: Compute covariance matrices in world space
    scales_matrix = torch.diag_embed(scales ** 2)  # [N, 3, 3]
    cov_world = R_gaussians @ scales_matrix @ R_gaussians.transpose(1, 2)  # [N, 3, 3]

    # Step 4: Transform covariance matrices to camera space
    R_cam = viewmat[:3, :3]  # [3, 3]
    R_cam_expanded = R_cam.unsqueeze(0).expand(N, 3, 3)
    cov_cam = R_cam_expanded @ cov_world @ R_cam_expanded.transpose(1, 2)  # [N, 3, 3]

    # Step 5: Project means onto the image plane
    means_proj_hom = (K @ means_cam.T).T  # [N, 3]
    means2D = means_proj_hom[:, :2] / means_proj_hom[:, 2:3]  # [N, 2]

    # Filter Gaussians that are within the view frustum
    z = means_cam[:, 2]  # [N]
    mask = (z > near_plane) & (z < far_plane)
    if not mask.any():
        print("No Gaussians are visible in the current view.")
        return None

    means2D = means2D[mask]
    cov_cam = cov_cam[mask]
    opacities = opacities[mask]
    colors = colors[mask]
    z = z[mask]
    N = means2D.size(0)  # Update N after masking

    # Step 6: Compute 2D covariance matrices using the Jacobian
    fx = K[0, 0]
    fy = K[1, 1]
    x = means_cam[mask, 0]
    y = means_cam[mask, 1]

    J = torch.zeros(N, 2, 3, device=device, dtype=dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / z**2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / z**2

    cov2D = J @ cov_cam @ J.transpose(1, 2)  # [N, 2, 2]

    # Compute radii for bounding boxes
    eigenvalues = torch.linalg.eigvalsh(cov2D)  # [N, 2]
    radii = 2 * torch.sqrt(eigenvalues.max(dim=1).values)  # [N]

    # Compute bounding rectangles
    x_min = (means2D[:, 0] - radii).clamp(0, width - 1)
    x_max = (means2D[:, 0] + radii).clamp(0, width - 1)
    y_min = (means2D[:, 1] - radii).clamp(0, height - 1)
    y_max = (means2D[:, 1] + radii).clamp(0, height - 1)

    rects = ((x_min, y_min), (x_max, y_max))  # Each is [N]

    # Prepare render buffers
    render_color = torch.zeros(height, width, 3, device=device, dtype=dtype)
    render_depth = torch.zeros(height, width, 1, device=device, dtype=dtype)
    render_alpha = torch.zeros(height, width, 1, device=device, dtype=dtype)

    # Prepare pixel grid
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    pix_coord = torch.stack([grid_x, grid_y], dim=-1).float()  # [height, width, 2]

    # Rasterization loop
    for h in range(0, height, tile_size):
        for w in range(0, width, tile_size):
            tile_pix_coord = pix_coord[h:h+tile_size, w:w+tile_size].reshape(-1, 2)  # [B, 2]

            # Check which Gaussians overlap with the tile
            over_tl_x = rects[0][0].clamp(min=w)
            over_tl_y = rects[0][1].clamp(min=h)
            over_br_x = rects[1][0].clamp(max=w+tile_size-1)
            over_br_y = rects[1][1].clamp(max=h+tile_size-1)

            in_mask = (over_br_x > over_tl_x) & (over_br_y > over_tl_y)

            if not in_mask.any():
                continue

            P = in_mask.sum()

            # Get the Gaussians in the tile
            means2D_in = means2D[in_mask]  # [P, 2]
            cov2D_in = cov2D[in_mask]      # [P, 2, 2]
            depths_in = z[in_mask]         # [P]
            opacities_in = opacities[in_mask]  # [P]
            colors_in = colors[in_mask]        # [P, 3]

            # Compute the inverse of covariance matrices
            cov2D_inv = torch.inverse(cov2D_in)  # [P, 2, 2]

            # Compute distances between pixels and Gaussian centers
            dx = tile_pix_coord[:, None, :] - means2D_in[None, :, :]  # [B, P, 2]

            # Compute Gaussian weights
            exponents = torch.einsum('bpi,pij,bpj->bp', dx, cov2D_inv, dx)  # [B, P]
            gauss_weight = torch.exp(-0.5 * exponents)  # [B, P]

            # Compute alpha values
            alpha = (gauss_weight * opacities_in[None, :]).clamp(max=0.99)  # [B, P]

            # Sort Gaussians by depth (front to back)
            sorted_depths, indices = depths_in.sort()
            gauss_weight = gauss_weight[:, indices]
            alpha = alpha[:, indices]
            colors_in = colors_in[indices]

            # Compute transmittance T and accumulated alpha
            T = torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1).cumprod(dim=1)  # [B, P]
            acc_alpha = (alpha * T).sum(dim=1, keepdim=True)  # [B, 1]

            # Compute accumulated color
            tile_color = (T[:, :, None] * alpha[:, :, None] * colors_in[None, :, :]).sum(dim=1)  # [B, 3]

            # Compute accumulated depth
            tile_depth = (T * alpha * sorted_depths[None, :]).sum(dim=1, keepdim=True)  # [B, 1]

            # Reshape to tile_size x tile_size
            B = tile_pix_coord.size(0)
            size_y = min(tile_size, height - h)
            size_x = min(tile_size, width - w)
            tile_color = tile_color.reshape(size_y, size_x, -1)
            tile_depth = tile_depth.reshape(size_y, size_x, -1)
            acc_alpha = acc_alpha.reshape(size_y, size_x, -1)

            # Update render buffers
            render_color[h:h+size_y, w:w+size_x] = tile_color
            render_depth[h:h+size_y, w:w+size_x] = tile_depth
            render_alpha[h:h+size_y, w:w+size_x] = acc_alpha

    return {
        "render": render_color,
        "depth": render_depth,
        "alpha": render_alpha
    }
