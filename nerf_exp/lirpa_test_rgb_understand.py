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

if __name__ == "__main__":
    view_mat = torch.Tensor([
        [-0.2569, -0.9664,  0.0042, -0.0599],
        [-0.4229,  0.1085, -0.8997, -0.1884],
        [ 0.8690, -0.2329, -0.4365,  0.4518],
        [ 0.0000,  0.0000,  0.0000,  1.0000]
    ])  # 4x4
    width = 16
    height = 16
    eps2d = 0.3
    tile_coord = torch.Tensor([[5,0]])

    eps = 0.0000

    perturb = torch.rand(view_mat.shape)*eps*2-eps

    perturbed_view_mat = view_mat+perturb

    mean = torch.Tensor([[-0.21954189240932465], [-0.000968221458606422], [-0.10691410303115845], [1]]) # 4x1
    quats = torch.Tensor([[0.48229265213012695, 0.8909245729446411, 0.15250934660434723, 0.4660235345363617]])  
    scales = torch.exp(torch.Tensor([-9.364758491516113, -7.122779846191406, -7.10692834854126]))
    cov_world = torch.Tensor([
        [4.154129555899999e-07, -2.3633104717646347e-07, -2.1989748688611144e-07], 
        [-2.3633104717646347e-07, 4.4870313331557554e-07, -1.9406923001952237e-07], 
        [-2.1989748688611144e-07, -1.9406923001952237e-07, 4.6513332563336007e-07]
    ]) # 3x3 
    K = torch.Tensor([
        [1.2000e+03, 0.0000e+00, 8.0000e+00],
        [0.0000e+00, 1.2000e+03, 8.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]
    ]) # 3x3
    opacity = torch.sigmoid(torch.Tensor([2.3424]))

    means_cam_hom = (view_mat@mean).transpose(0,1) # 1x4
    means_cam = means_cam_hom[:,:3]/ means_cam_hom[:,3:4] # 1x3

    # R_gaussians = quaternion_to_rotation_matrix(quats)
    # scales_matrix = torch.diag_embed(scales)  # [N, 3, 3]
    # M = R_gaussians@scales_matrix
    # tmp = M@M.transpose(0,1)

    R_cam = view_mat[:3, :3] # 3x3 
    cov_cam = R_cam@cov_world@R_cam.transpose(0,1) # 3x3

    means_proj_hom = (K@means_cam.T).T
    means2D = means_proj_hom[:,:2]/means_proj_hom[:,2:3]
    
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

    J = torch.zeros(2, 3) 
    J[0, 0] = fx / z_cam
    J[0, 2] = -fx * tx / z_cam**2
    J[1, 1] = fy / z_cam
    J[1, 2] = -fy * ty / z_cam**2

    cov2D = J @ cov_cam @ J.transpose(0, 1)  # 2x2
    det_orig = cov2D.det()
    cov2D[0,0] = cov2D[0,0]+eps2d 
    cov2D[1,1] = cov2D[1,1]+eps2d 
    det_blur = cov2D.det()
    det = det_blur 
    dx = tile_coord - means2D 
    conic = cov2D.inverse()

    gauss_weight = torch.exp(
        -0.5 * (
        dx[:,0]**2 * conic[0,0] 
        + dx[:,1]**2 * conic[1,1]
        + dx[:,0]*dx[:,1] * conic[0,1]
        + dx[:,0]*dx[:,1] * conic[1,0])
    )
    print(gauss_weight)
    