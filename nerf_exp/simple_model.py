import torch 
import numpy as np 

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
    
class RasterizationModelRGBManual_notile(torch.nn.Module):
    def __init__(
        self,
        data_pack,
        fx=2343.0242837919386,
        fy=2343.0242837919386,
        width=2560,
        height=1440,
        tile_size=16,
    ):
        super().__init__()  # Initialize the base class first

        self.fx = fx 
        self.fy = fy
        self.width = width
        self.height = height

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.setup_camera(data_pack)
        self.prepare_rasterization_coefficients(
            tile_size=tile_size
        )
        # self.shrink_rasterization_coefficients()

        self.prepare_render_coefficients()
        self.colors = None 

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(RasterizationModelRGBManual_notile, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move more custom tensors
            self.means_hom = self.means_hom.to(device)
            self.means_hom_tmp = self.means_hom_tmp.to(device)
            self.cov_world = self.cov_world.to(device)
            self.means = self.means.to(device)
            self.quats = self.quats.to(device)
            self.scales = self.scales.to(device)
            self.opacities = self.opacities.to(device)
            self.opacities_rast = self.opacities_rast.to(device)
            # self.overall_mask = self.overall_mask.to(device)  # Avoid error

            self.K = self.K.to(device)
            self.lim_x = self.lim_x.to(device) 
            self.lim_y = self.lim_y.to(device)
            self.tile_coord = self.tile_coord.to(device)
            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining

    def setup_camera(self, data_pack):
        self.opacities = data_pack['opacities'].to(self.device)
        self.means = data_pack['means'].to(self.device)
        # base_color = self.model.base_colors
        self.scales = data_pack['scales'].to(self.device)
        self.quats = data_pack['quats'].to(self.device)

        # apply the compensation of screen space blurring to gaussians
        self.BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.K = torch.Tensor([
            [self.fx, 0, self.width/2],
            [0, self.fy, self.height/2],
            [0,0,1]
        ]).to(self.device)

    @torch.no_grad()
    def prepare_rasterization_coefficients(
            self,
            near_plane=0.01, 
            far_plane=1e10, 
            sh_degree=3, 
            tile_size=16,
            eps2d = 0.3,
            radius_clip = 0.0
        ):
        self.tile_size = tile_size

        Ks = torch.tensor([[
            [self.fx, 0, self.width/2],
            [0, self.fy, self.height/2],
            [0, 0, 1]
        ]]).to(self.device)
        means = self.means 
        quats = self.quats 
        scales = torch.exp(self.scales)
        opacities = torch.sigmoid(self.opacities)
        self.opacities_rast = opacities[None, None,...]
        width = self.width
        height = self.height
        
        K = Ks[0]              # [3, 3]

        N = means.size(0)
        self.N = N
    
        dtype = means.dtype

        # Step 1: Transform Gaussian centers to camera space
        ones = torch.ones(N, 1, device=self.device, dtype=dtype)
        self.means_hom = torch.cat([means, ones], dim=1).to(self.device)  # [N, 4]
        self.means_hom = self.means_hom.unsqueeze(0)
        self.means_hom_tmp = self.means_hom.transpose(1,2)
        
        # Step 2: Compute rotation matrices from quaternions
        R_gaussians = quaternion_to_rotation_matrix(quats)  # [N, 3, 3]

        # Step 3: Compute covariance matrices in world space
        scales_matrix = torch.diag_embed(scales).to(self.device)  # [N, 3, 3]
        M = R_gaussians@scales_matrix
        self.cov_world = M @ M.transpose(1, 2)  # [N, 3, 3]
        # self.cov_world = self.cov_world.unsqueeze(0)  # Don't need an extra dimension that brings confusions
        self.tan_fovx = 0.5*self.width/self.fx 
        self.tan_fovy = 0.5*self.height/self.fy 
        self.lim_x = torch.Tensor([1.3*self.tan_fovx]).to(self.device) 
        self.lim_y = torch.Tensor([1.3*self.tan_fovy]).to(self.device)

    @torch.no_grad()
    def prepare_render_coefficients(self):
        pix_coord = torch.stack(torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing='xy'), dim=-1).to(self.device)
        self.tile_coord = pix_coord.flatten(0,-2)[None,:,None,:]

    def forward(self, x):
        means_cam_hom = torch.matmul(x, self.means_hom_tmp).transpose(1,2)    # [N, 4]
        means_cam = means_cam_hom[:, :, :3] / means_cam_hom[:, :, 3:4]  # [N, 3]

        R_cam = x[:, :3, :3]  # [1, 3, 3]
        R_cam = R_cam.unsqueeze(1)  # Add an extra dimension for broadcasting
        # First multiplication: R_cam @ self.cov_world
        cov_temp = torch.matmul(R_cam, self.cov_world)  # Shape: [1, N, 3, 3]
        # Second multiplication: result @ R_cam.transpose(-1, -2)
        cov_cam = torch.matmul(cov_temp, R_cam.transpose(-1, -2))  # Shape: [1, N, 3, 3]

        # Step 5: Project means onto the image plane
        means_proj_hom = means_cam @ self.K.t()
        means2D = means_proj_hom[:, :, :2] / means_proj_hom[:, :, 2:3]  # [N, 2]

        # # Step 6: Compute 2D covariance matrices using the Jacobian
        x = means_cam[:, :, 0]
        y = means_cam[:, :, 1]
        z_cam = means_cam[:, :, 2]

        tx = z_cam*torch.min(self.lim_x, torch.max(-self.lim_x, x/z_cam))
        ty = z_cam*torch.min(self.lim_y, torch.max(-self.lim_y, y/z_cam))

        J00 = self.fx / z_cam
        J02 = -self.fx * tx / z_cam**2
        J11 = self.fy / z_cam
        J12 = -self.fy * ty / z_cam**2

        cov2D00 = (
            J00 * J00 * cov_cam[:,:,0, 0] +
            J00 * J02 * cov_cam[:,:,0, 2] +
            J02 * J00 * cov_cam[:,:,2, 0] +
            J02 * J02 * cov_cam[:,:,2, 2]
        )
        
        # Compute C[1][1]
        cov2D11 = (
            J11 * J11 * cov_cam[:,:,1, 1] +
            J11 * J12 * cov_cam[:,:,1, 2] +
            J12 * J11 * cov_cam[:,:,2, 1] +
            J12 * J12 * cov_cam[:,:,2, 2]
        )
        
        # Compute C[0][1]
        cov2D01 = (
            J00 * J11 * cov_cam[:,:,0, 1] +
            J00 * J12 * cov_cam[:,:,0, 2] +
            J02 * J11 * cov_cam[:,:,2, 1] +
            J02 * J12 * cov_cam[:,:,2, 2]
        )
        
        # Compute C[1][0]
        cov2D10 = (
            J11 * J00 * cov_cam[:,:,1, 0] +
            J11 * J02 * cov_cam[:,:,1, 2] +
            J12 * J00 * cov_cam[:,:,2, 0] +
            J12 * J02 * cov_cam[:,:,2, 2]
        )
        cov2D00 = cov2D00+0.3
        cov2D11 = cov2D11+0.3 
        det = cov2D00*cov2D11-cov2D01*cov2D10

        b = 0.5*(cov2D00+cov2D11)
        v1 = b+torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
        v2 = b-torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
        radius = 3*torch.sqrt(torch.max(v1, v2))

        # conic is the inverse of cov2d
        conic00 = 1/(cov2D00*cov2D11-cov2D01*cov2D10)*cov2D11
        conic01 = 1/(cov2D00*cov2D11-cov2D01*cov2D10)*(-cov2D01)
        conic10 = 1/(cov2D00*cov2D11-cov2D01*cov2D10)*(-cov2D10)
        conic11 = 1/(cov2D00*cov2D11-cov2D01*cov2D10)*cov2D00

        dx = self.tile_coord-means2D[:,None,:]

        gauss_weight_orig = torch.exp(-0.5 * (
            dx[:,:,:,0]**2 * conic00[:, None, :] 
            + dx[:,:,:,1]**2 * conic11[:, None, :]
            + dx[:,:,:,0]*dx[:,:,:,1] * conic01[:, None, :]
            + dx[:,:,:,0]*dx[:,:,:,1] * conic10[:, None, :]))
        
        alpha = gauss_weight_orig[:,:,:,None]*self.opacities_rast

        # alpha_clip = torch.clip(alpha, max=0.99)
        alpha_clip = -torch.nn.functional.relu(-alpha+0.99)+0.99

        # depth_order = torch.argsort(means_cam[:,:,2], dim=1).squeeze()
        # sorted_alpha = alpha_clip[:,:,depth_order,:]
        # sorted_T = torch.cat([torch.ones_like(sorted_alpha[:,:,:1]), 1-sorted_alpha[:,:,:-1]], dim=2).cumprod(dim=2)

        return alpha_clip

class DepthModel(torch.nn.Module):
    def __init__(
            self,
            input_model: RasterizationModelRGBManual_notile, 

    ):
        super(DepthModel, self).__init__()
        self.means_hom_tmp = input_model.means_hom_tmp
        self.device = input_model.device 

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(DepthModel, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            self.means_hom_tmp = self.means_hom_tmp.to(device)
            self.device = torch.device(device)
        return self         

    def forward(self, x):      
        means_cam_hom = torch.matmul(x, self.means_hom_tmp).transpose(1,2)    # [N, 4]
        means_cam = means_cam_hom[:, :, :3] / means_cam_hom[:, :, 3:4]  # [N, 3]

        return means_cam[:,:,2]
    
class RGBModel(torch.nn.Module):
    def forward(self, x):
        pass 