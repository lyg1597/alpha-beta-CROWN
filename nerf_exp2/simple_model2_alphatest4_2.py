import torch 
from inverse_op3 import Inverse

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
    
class AlphaModel(torch.nn.Module):
    def __init__(
        self,
        data_pack,
        fx=2343.0242837919386,
        fy=2343.0242837919386,
        width=2560,
        height=1440,
        tile_coord = None,
        method = 'crown',
    ):
        super().__init__()  # Initialize the base class first

        self.fx = fx 
        self.fy = fy
        self.width = width
        self.height = height
        self.inv_op = Inverse()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.setup_camera(data_pack)
        self.prepare_rasterization_coefficients()
        # self.shrink_rasterization_coefficients()

        self.prepare_render_coefficients(tile_coord)
        self.colors = None 

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(AlphaModel, self).to(*args, **kwargs)
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
        ):

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
    def prepare_render_coefficients(self, tile_coord):
        if tile_coord is None:
            pix_coord = torch.stack(torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing='xy'), dim=-1).to(self.device)
            self.tile_coord = pix_coord.flatten(0,-2)[None,:,:]
        else:
            self.tile_coord = tile_coord[None,:,:].to(self.device)

    def forward(self, means_hom_tmp, x, cov_world, opacities_rast):
        # # Define your computation here.
        # gamma = x[:,0:1]
        # beta = x[:,1:2]
        # alpha = x[:,2:3]
        # R00 = torch.cos(alpha)*torch.cos(beta)
        # R01 = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma)-torch.sin(alpha)*torch.cos(gamma)
        # R02 = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma)+torch.sin(alpha)*torch.sin(gamma)
        # R03 = x[:,3:4]
        # R10 = torch.sin(alpha)*torch.cos(beta)
        # R11 = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma)+torch.cos(alpha)*torch.cos(gamma)
        # R12 = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma)-torch.cos(alpha)*torch.sin(gamma)
        # R13 = x[:,4:5]
        # R20 = -torch.sin(beta)
        # R21 = torch.cos(beta)*torch.sin(gamma)
        # R22 = torch.cos(beta)*torch.cos(gamma)
        # R23 = x[:,5:6]
        # combined = torch.cat([R00, R01, R02, R03, R10, R11, R12, R13, R20, R21, R22, R23], dim=1)
        # result = combined.view(-1, 3, 4)
        # # 3) Prepare the fixed 4th row [0, 0, 0, 1] as shape [N, 1, 4]
        # #    We'll broadcast (expand) this row for each of the N samples.
        # fixed_row = torch.tensor([0, 0, 0, 1]).view(1, 1, 4).to(self.device)
        # fixed_row = fixed_row.expand(result.shape[0], 1, 4)  # shape: [N, 1, 4]

        # # 4) Concatenate the top 3 rows and the fixed 4th row => [N, 4, 4]
        # x = torch.cat([result, fixed_row], dim=1)  # shape: [N, 4, 4]

        means_cam_hom = torch.matmul(means_hom_tmp.transpose(1,2), x.transpose(1,2))    # [N, 4]
        means_cam = means_cam_hom[:,:,:3]
        # means_cam = means_cam_hom[:, :, :3] / means_cam_hom[:, :, 3:4]  # [N, 3]

        # Step 5: Project means onto the image plane
        means_proj_hom = means_cam @ self.K.t()
        z0 = means_proj_hom[:,:,2:3]
        # means2D = means_proj_hom[:, :, :2] / means_proj_hom[:, :, 2:3]  # [N, 2]
        means2D = means_proj_hom[:, :, :2]
        # return means_proj_hom 
    
        R_cam = x[:, :3, :3]  # [1, 3, 3]
        # R_cam = R_cam.unsquSAeeze(1)  # Add an extra dimension for broadcasting
        # First multiplication: R_cam @ self.cov_world
        cov_temp = torch.matmul(R_cam, cov_world)  # Shape: [1, N, 3, 3]
        # Second multiplication: result @ R_cam.transpose(-1, -2)
        cov_cam = torch.matmul(cov_temp, R_cam.transpose(-1, -2))  # Shape: [1, N, 3, 3]
        
        # # Step 6: Compute 2D covariance matrices using the Jacobian
        x = means_cam[:, :, 0]
        y = means_cam[:, :, 1]
        z_cam = means_cam[:, :, 2]

        # tx = torch.min(z_cam*self.lim_x, torch.max(-z_cam*self.lim_x, x))
        # ty = torch.min(z_cam*self.lim_y, torch.max(-z_cam*self.lim_y, y))
        # return ty

        J00 = z_cam*self.fx 
        J02 = -self.fx * x 
        J11 = z_cam*self.fy 
        J12 = -self.fy * y 
        # J00 = self.fx / z_cam
        # J02 = -self.fx * tx / z_cam**2
        # J11 = self.fy / z_cam
        # J12 = -self.fy * ty / z_cam**2
        # return J12

        cov2D00 = (
            J00 * J00 * cov_cam[:,0:1,0] +
            J00 * J02 * cov_cam[:,0:1,2] +
            J02 * J00 * cov_cam[:,2:3,0] +
            J02 * J02 * cov_cam[:,2:3,2]
        )
        
        # Compute C[1][1]
        cov2D11 = (
            J11 * J11 * cov_cam[:,1:2,1] +
            J11 * J12 * cov_cam[:,1:2,2] +
            J12 * J11 * cov_cam[:,2:3,1] +
            J12 * J12 * cov_cam[:,2:3,2]
        )
        
        # Compute C[0][1]
        cov2D0110 = (
            J00 * J11 * cov_cam[:,0:1,1] +
            J00 * J12 * cov_cam[:,0:1,2] +
            J02 * J11 * cov_cam[:,2:3,1] +
            J02 * J12 * cov_cam[:,2:3,2]
        )
    
        # # Compute C[1][0]
        # cov2D10 = (
        #     J11 * J00 * cov_cam[:,:,1, 0] +
        #     J11 * J02 * cov_cam[:,:,1, 2] +
        #     J12 * J00 * cov_cam[:,:,2, 0] +
        #     J12 * J02 * cov_cam[:,:,2, 2]
        # )
        cov2D00 = cov2D00 
        cov2D11 = cov2D11
        
        cov2D = torch.stack([cov2D00[:,:], cov2D0110[:,:], cov2D0110[:,:], cov2D11[:,:]], dim=2).reshape((-1,2,2))
        conic = self.inv_op(cov2D)
        conic00 = conic[:,0,0]
        conic0110 = conic[:,0,1]
        conic11 = conic[:,1,1]
   
        dx = z0*z0*self.tile_coord-z0*means2D
        inside = (
            torch.square(dx[:,:,0]) * conic00[:, None] 
            + torch.square(dx[:,:,1]) * conic11[:, None]
            + dx[:,:,0]*dx[:,:,1] * conic0110[:, None]
            + dx[:,:,0]*dx[:,:,1] * conic0110[:, None])
        # inside = -0.5 * 
        return inside
        gauss_weight_orig = torch.exp(inside)
        alpha = gauss_weight_orig*opacities_rast
        return alpha

        # alpha_clip = torch.clip(alpha, max=0.99)
        alpha_clip = -torch.nn.functional.relu(-alpha+0.99)+0.99

        return alpha_clip


class DepthModel(torch.nn.Module):
    def __init__(
            self,
            input_model: AlphaModel, 

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

    def forward(self, means_hom_tmp, x):
        means_cam_hom = torch.matmul(means_hom_tmp.transpose(1,2), x.transpose(1,2))    # [N, 4]
        means_cam = means_cam_hom[:, :, :3]

        return means_cam[:,:,2]

class MeanModel(torch.nn.Module):
    def __init__(
        self,
        means,
        scales,
        quats,
        fx=2343.0242837919386,
        fy=2343.0242837919386,
        width=2560,
        height=1440,
    ):
        super().__init__()  # Initialize the base class first

        self.fx = fx 
        self.fy = fy
        self.width = width
        self.height = height
        self.inv_op = Inverse()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.means = means.to(self.device)
        self.scales = scales.to(self.device)
        self.quats = quats.to(self.device)
        # base_color = self.model.base_colors

        # apply the compensation of screen space blurring to gaussians
        self.K = torch.Tensor([
            [self.fx, 0, self.width/2],
            [0, self.fy, self.height/2],
            [0,0,1]
        ]).to(self.device)
        self.prepare_rasterization_coefficients()

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(MeanModel, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move more custom tensors
            self.means_hom = self.means_hom.to(device)
            self.means_hom_tmp = self.means_hom_tmp.to(device)
            self.means = self.means.to(device)
            self.quats = self.quats.to(device)
            self.scales = self.scales.to(device)
            self.cov_world = self.cov_world.to(device)
            # self.overall_mask = self.overall_mask.to(device)  # Avoid error

            self.K = self.K.to(device)
            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining

    @torch.no_grad()
    def prepare_rasterization_coefficients(
            self,
        ):
        Ks = self.K[None]
        means = self.means 
        quats = self.quats 
        scales = torch.exp(self.scales)
        
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

    def get_radii(self, x, cp):
        means_cam_hom = torch.matmul(x.transpose(1,2), cp.transpose(1,2))    # [N, 4]
        # means_cam = means_cam_hom[:,:,:3]
        means_cam = means_cam_hom[:, :, :3] / means_cam_hom[:, :, 3:4]  # [N, 3]

        # Step 5: Project means onto the image plane
        means_proj_hom = means_cam @ self.K.t()
        # means2D = means_proj_hom[:, :, :2]
    
        R_cam = cp[:, :3, :3]  # [1, 3, 3]
        R_cam = R_cam  # Add an extra dimension for broadcasting
        # First multiplication: R_cam @ self.cov_world
        cov_temp = torch.matmul(R_cam, self.cov_world)  # Shape: [1, N, 3, 3]
        # Second multiplication: result @ R_cam.transpose(-1, -2)
        cov_cam = torch.matmul(cov_temp, R_cam.transpose(-1, -2))  # Shape: [1, N, 3, 3]
        
        # # Step 6: Compute 2D covariance matrices using the Jacobian
        x = means_cam[:, :, 0]
        y = means_cam[:, :, 1]
        z_cam = means_cam[:, :, 2]

        # tx = torch.min(z_cam*self.lim_x, torch.max(-z_cam*self.lim_x, x))
        # ty = torch.min(z_cam*self.lim_y, torch.max(-z_cam*self.lim_y, y))
        # return ty

        # J00 = z_cam*self.fx 
        # J02 = -self.fx * x 
        # J11 = z_cam*self.fy 
        # J12 = -self.fy * y 
        J00 = self.fx / z_cam
        J02 = -self.fx * x / z_cam**2
        J11 = self.fy / z_cam
        J12 = -self.fy * y / z_cam**2
        # return J12

        cov2D00 = (
            J00 * J00 * cov_cam[:,0:1, 0] +
            J00 * J02 * cov_cam[:,0:1, 2] +
            J02 * J00 * cov_cam[:,2:3, 0] +
            J02 * J02 * cov_cam[:,2:3, 2]
        )
        
        # Compute C[1][1]
        cov2D11 = (
            J11 * J11 * cov_cam[:,1:2, 1] +
            J11 * J12 * cov_cam[:,1:2, 2] +
            J12 * J11 * cov_cam[:,2:3, 1] +
            J12 * J12 * cov_cam[:,2:3, 2]
        )
        
        # Compute C[0][1]
        cov2D0110 = (
            J00 * J11 * cov_cam[:,0:1, 1] +
            J00 * J12 * cov_cam[:,0:1, 2] +
            J02 * J11 * cov_cam[:,2:3, 1] +
            J02 * J12 * cov_cam[:,2:3, 2]
        )
    
        # # Compute C[1][0]
        # cov2D10 = (
        #     J11 * J00 * cov_cam[:,:,1, 0] +
        #     J11 * J02 * cov_cam[:,:,1, 2] +
        #     J12 * J00 * cov_cam[:,:,2, 0] +
        #     J12 * J02 * cov_cam[:,:,2, 2]
        # )
        cov2D00 = cov2D00+0.003 
        cov2D11 = cov2D11+0.003
        
        det = cov2D00*cov2D11-cov2D0110*cov2D0110

        b = 0.5*(cov2D00+cov2D11)
        v1 = b+torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
        v2 = b-torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
        radius = torch.ceil(3*torch.sqrt(torch.max(v1, v2)))
        return radius 

    def forward(self, x, cp):
        means_cam_hom = torch.matmul(x.transpose(1,2), cp.transpose(1,2))    # [N, 4]
        # means_cam = means_cam_hom[:,:,:3]
        means_cam = means_cam_hom[:, :, :3] 

        # Step 5: Project means onto the image plane
        means_proj_hom = means_cam @ self.K.t()
        # z0 = means_proj_hom[:,:,2:3]
        # means2D = means_proj_hom[:, :, :2]
        return means_proj_hom
            