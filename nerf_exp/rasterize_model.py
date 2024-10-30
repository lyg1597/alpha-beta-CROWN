
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
from pathlib import Path 
import yaml 
from splatfactoenv.splatfactoenv_model import SplatfactoEnvModel
from nerfstudio.data.scene_box import SceneBox
import numpy as np 
from nerfstudio.cameras.cameras import Cameras, CameraType
import matplotlib.pyplot as plt 
try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
    
from test2 import rasterize_gaussians_pytorch, render, render_notile
from gsplat.rendering import rasterization
@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

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

class RasterizationModel(torch.nn.Module):
    def __init__(
        self,
        output_folder, 
        camera_pose,
        fx=2343.0242837919386,
        fy=2343.0242837919386,
        width=2560,
        height=1440,
        camera_type=CameraType.PERSPECTIVE,
        tile_size=16,
    ):
        super().__init__()  # Initialize the base class first

        self.fx = fx 
        self.fy = fy
        self.width = width
        self.height = height
        self.camera_type = camera_type

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_folder = output_folder 
        self.config_path = os.path.join(output_folder, 'config.yml')
        self.load_path = os.path.join(output_folder, './nerfstudio_models/step-000089999.ckpt')
        self.config_path = Path(self.config_path)

        self.config = yaml.load(self.config_path.read_text(), Loader=yaml.Loader)

        metadata={
            'depth_filenames': None, 
            'depth_unit_scale_factor': 0.001, 
            'mask_color': None,
            "env_params":torch.tensor([[1.0,0.0]])
        }

        self.model:SplatfactoEnvModel = self.config.pipeline.model.setup(
            scene_box = SceneBox(
                    aabb=torch.Tensor([
                        [-1., -1., -1.],
                        [ 1.,  1.,  1.]
                    ]),
            ),
            num_train_data = 1440,
            metadata = metadata, 
            device = self.device,
            grad_scaler = None, 
            seed_points = None 
        )
        self.model.training = False
        self.model.to(self.device)

        loaded_state = torch.load(self.load_path, map_location='cuda')
        self.load_state_dict_modify(loaded_state)

        self.transform = np.array([
            [
                0.9996061325073242,
                -0.01975083164870739,
                -0.01993674598634243,
                997.8822021484375
            ],
            [
                -0.01975083164870739,
                0.009563744068145752,
                -0.9997591972351074,
                -42.26317596435547
            ],
            [
                0.01993674598634243,
                0.9997591972351074,
                0.00916987657546997,
                -242.0419158935547
            ]
        ])

        self.scale_factor = 0.0003946526873285077

        camera_pose = self.transform@camera_pose
        camera_pose[:3,3] *= self.scale_factor
        camera_pose = camera_pose[:3,:]
        cam_state = camera_pose

        if cam_state.ndim == 2:
            cam_state = np.expand_dims(cam_state, axis=0)

        self.camera_to_world = torch.FloatTensor( cam_state ).to(self.device)

        self.setup_camera()
        self.prepare_rasterization_coefficients(
            tile_size=tile_size
        )
        # self.shrink_rasterization_coefficients()

        # self.prepare_render_coefficients()

    # def shrink_rasterization_coefficients(self):
    #     pass 

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(RasterizationModel, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move custom tensors
            self.sh_coeffs_expanded = self.sh_coeffs_expanded.to(device)
            self.overall_mask = self.overall_mask.to(device)
            self.means_cam_rast = self.means_cam_rast.to(device)
            self.cov2D_rast = self.cov2D_rast.to(device)  
            self.means2D_rast = self.means2D_rast.to(device)
            self.opacities_rast = self.opacities_rast.to(device)
            self.radius = self.radius.to(device)
            
            # # Render coefficients
            # self.render_color = self.render_color.to(device)
            # self.render_depth = self.render_depth.to(device)
            # self.render_alpha = self.render_alpha.to(device)
            # for item in self.precomputed_tiles:
            #     for key, vals in item.items():
            #         if type(vals) == torch.Tensor:
            #             item[key] = vals.to(device)
            
            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining

    def load_state_dict_modify(self, loaded_state):

        step = loaded_state['step']
        loaded_state = loaded_state['pipeline']
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)

        state_dict = state 
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}
        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            self.model.load_state_dict(model_state, strict=False)        

    def setup_camera(self):
        self.camera = Cameras(
            camera_to_worlds=self.camera_to_world, 
            fx=self.fx, 
            fy=self.fy,
            cx=self.width/2,
            cy=self.height/2,
            width=self.width,
            height=self.height,
            distortion_params=None,
            camera_type=self.camera_type, 
            metadata=None
        )

        if not isinstance(self.camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # if camera.metadata is None:
        #     camera.metadata = {"env_prams":torch.tensor([0.5,0.5])}

        if self.model.config.sh_degree > 0:
            self.sh_degree_to_use = min(self.model.step // self.model.config.sh_degree_interval, self.model.config.sh_degree)

        if self.model.training:
            # if not camera.shape[0] == 1:
            #     print(">>>>>>>>>>>", camera.shape[0])
            # val = camera.shape[0]
            # string = "Only one camera at a time, received " + str(val)
            # assert camera.shape[0] == 1, string
            optimized_camera_to_world = self.model.camera_optimizer.apply_to_camera(self.camera)
        else:
            optimized_camera_to_world = self.camera.camera_to_worlds


        camera_scale_fac = self.model._get_downscale_factor()
        self.camera.rescale_output_resolution(1 / camera_scale_fac)
        self.viewmat = get_viewmat(optimized_camera_to_world)
        self.W, self.H = int(self.camera.width.item()), int(self.camera.height.item())
        self.model.last_size = (self.H, self.W)
        # camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        self.opacities = self.model.opacities
        self.means = self.model.means
        # base_color = self.model.base_colors
        self.scales = self.model.scales
        self.quats = self.model.quats
        
        # apply the compensation of screen space blurring to gaussians
        self.BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.K = self.camera.get_intrinsics_matrices().cuda()
        if self.model.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.model.config.rasterize_mode)

        if self.model.config.output_depth_during_training or not self.training:
            self.render_mode = "RGB+ED"
        else:
            self.render_mode = "RGB"

        # if camera.metadata is None:
        #     env_params = torch.Tensor([0.5,0.5])
        # else:
        #     env_params = camera.metadata['env_params']
        # env_params_repeat = env_params.repeat(means.shape[0], 1).to(self.device)

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

        view_mats =  get_viewmat(self.camera.camera_to_worlds).to(self.means.device)
        Ks = torch.tensor([[
            [self.camera.fx, 0, self.camera.cx],
            [0, self.camera.fy, self.camera.cy],
            [0, 0, 1]
        ]]).to(self.means.device)
        means = self.means 
        quats = self.quats 
        scales = torch.exp(self.scales)
        opacities = torch.sigmoid(self.opacities)
        width = self.width
        height = self.height
        
        viewmat = view_mats[0]  # [4, 4]
        K = Ks[0]              # [3, 3]

        N = means.size(0)
    
        dtype = means.dtype

        # Step 1: Transform Gaussian centers to camera space
        ones = torch.ones(N, 1, device=self.device, dtype=dtype)
        means_hom = torch.cat([means, ones], dim=1).to(self.device)  # [N, 4]
        means_cam_hom = (viewmat @ means_hom.T).T    # [N, 4]
        means_cam = means_cam_hom[:, :3] / means_cam_hom[:, 3:4]  # [N, 3]

        mask = (means_cam[:,2]>near_plane) & (means_cam[:,2]<far_plane)
        self.overall_mask = mask
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
        scales_matrix = torch.diag_embed(scales).to(self.device)  # [N, 3, 3]
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

        J = torch.zeros(N, 2, 3, device=self.device, dtype=dtype)
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
        self.overall_mask = self.overall_mask & mask 
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
        self.overall_mask = self.overall_mask & mask 
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
        self.overall_mask = self.overall_mask & mask 
        
        # # mask out opacities smaller than threshold
        # threshold, _ = torch.kthvalue(opacities.squeeze(), opacities.shape[0]-300000+1)
        # mask = opacities>=threshold 
        # self.overall_mask = self.overall_mask & mask.squeeze()
        
        means = means[self.overall_mask]
        self.means_cam_rast = means_cam[self.overall_mask]
        self.cov2D_rast = cov2D[self.overall_mask]
        self.means2D_rast = means2D[self.overall_mask]
        self.opacities_rast = opacities[self.overall_mask]
        # colors = colors[self.overall_mask]
        self.radius = radius[self.overall_mask]
        N = means2D.size(0)  # Update N after masking
        
        means3D = means
        c2w = viewmat
        active_sh_degree = 3
        camtoworlds = torch.linalg.inv(c2w)
        rays_o = camtoworlds[:3,3]
        rays_d = means3D - rays_o
        rays_d = rays_d/rays_d.norm(dim=1, keepdim=True)
        sh_coeffs = eval_sh_old(active_sh_degree, rays_d)
        self.sh_coeffs_expanded = sh_coeffs.unsqueeze(2)  # Shape: (424029, 16, 1)

    @torch.no_grad()
    def prepare_render_coefficients(self):
        means2D=self.means2D_rast
        cov2D=self.cov2D_rast
        radii = self.radius
        opacity=self.opacities_rast
        depths = self.means_cam_rast[:,2]
        W=self.width
        H=self.height
        device=self.device
        tile_size=self.tile_size

        pix_coord = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(self.device)
        # radii = get_radius(cov2D)
        rect = get_rect(means2D, radii, width=W, height=H)

        self.render_color = torch.ones(*pix_coord.shape[:2], 3).to(device)
        self.render_depth = torch.zeros(*pix_coord.shape[:2], 1).to(device)
        self.render_alpha = torch.zeros(*pix_coord.shape[:2], 1).to(device)

        precomputed_tiles = []
        
        for h in range(0, H, tile_size):
            for w in range(0, W, tile_size):
                # Determine if the Gaussian overlaps with the current tile
                over_tl_x = rect[0][..., 0].clamp(min=w)
                over_tl_y = rect[0][..., 1].clamp(min=h)
                over_br_x = rect[1][..., 0].clamp(max=w+tile_size-1)
                over_br_y = rect[1][..., 1].clamp(max=h+tile_size-1)
                in_mask = (over_br_x > over_tl_x) & (over_br_y > over_tl_y)
                
                if not in_mask.any():
                    continue  # Skip tiles with no overlapping Gaussians
                
                # Prepare tile data
                tile_h = min(tile_size, H - h)
                tile_w = min(tile_size, W - w)
                tile_coord = pix_coord[h:h+tile_h, w:w+tile_w].reshape(-1, 2)
                
                indices_in_mask = in_mask.nonzero(as_tuple=True)[0]
                depths_in_mask = depths[indices_in_mask]
                sorted_depths, index = torch.sort(depths_in_mask)
                sorted_indices = indices_in_mask[index]  # Indices into the original Gaussians
                
                # Precompute sorted Gaussians and their properties
                sorted_means2D = means2D[sorted_indices]
                sorted_cov2D = cov2D[sorted_indices]
                sorted_conic = torch.inverse(sorted_cov2D)
                sorted_opacity = opacity[sorted_indices]
                
                # Compute distances
                dx = tile_coord[:, None, :] - sorted_means2D[None, :, :]  # Shape: [B, P, 2]
                
                # Compute Gaussian weights
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
                # Compute alpha values
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1).cumprod(dim=1)
                T_alpha = T * alpha  # Shape: [B, P]
                acc_alpha = T_alpha.sum(dim=1)  # Shape: [B]
                tile_depth = (T_alpha * sorted_depths[None, :]).sum(dim=1)  # Shape: [B]
                
                # Store precomputed data for this tile
                precomputed_tile = {
                    'h': h,
                    'w': w,
                    'tile_h': tile_h,
                    'tile_w': tile_w,
                    'T_alpha': T_alpha,            # For computing colors later
                    'acc_alpha': acc_alpha,        # Accumulated alpha
                    'tile_depth': tile_depth,      # Depth values
                    'sorted_indices': sorted_indices,  # Indices to retrieve colors later
                }
                precomputed_tiles.append(precomputed_tile)
        
        # precomputed_data = {
        #     'H': H,
        #     'W': W,
        #     'precomputed_tiles': precomputed_tiles,
        # }
        self.precomputed_tiles = precomputed_tiles

    # def forward(self, x):
    #     res = self.forward_old(x)
    #     return res 
    
    def forward_old(self, x):
        view_mats =  get_viewmat(self.camera.camera_to_worlds).to(self.means.device)
        Ks = torch.tensor([[
            [self.camera.fx, 0, self.camera.cx],
            [0, self.camera.fy, self.camera.cy],
            [0, 0, 1]
        ]]).to(self.means.device)
        res = rasterize_gaussians_pytorch(
            self.means,
            self.quats/self.quats.norm(dim=-1, keepdim=True),
            torch.exp(self.scales),
            torch.sigmoid(self.opacities),
            x, 
            view_mats,
            Ks,
            self.width,
            self.height
        )
        # pass
        return res  

    def forward(self, x):
        # shs = x[self.overall_mask]
        product = self.sh_coeffs_expanded*x 
        color = torch.sum(product, dim=1)
        color_rgb = (color+0.5).clip(min=0.0)
        # return color

        res = render(
            means2D=self.means2D_rast,
            cov2D=self.cov2D_rast,
            radii = self.radius,
            color=color_rgb,
            opacity=self.opacities_rast,
            depths = self.means_cam_rast[:,2],
            W=self.width,
            H=self.height,
            device=self.device,
            tile_size=self.tile_size
        )
        return res['render']

        # for tile_data in self.precomputed_tiles:
        #     h = tile_data['h']
        #     w = tile_data['w']
        #     tile_h = tile_data['tile_h']
        #     tile_w = tile_data['tile_w']
        #     T_alpha = tile_data['T_alpha']  # Shape: [B, P]
        #     # acc_alpha = tile_data['acc_alpha']  # Shape: [B]
        #     # tile_depth = tile_data['tile_depth']  # Shape: [B]
        #     sorted_indices = tile_data['sorted_indices']  # Indices into color
            
        #     # Retrieve colors for the Gaussians
        #     sorted_color = color_rgb[sorted_indices]  # Shape: [P, 3]
            
        #     # Compute the tile's color
        #     tile_color = (T_alpha * sorted_color[None]).sum(dim=1)  # Shape: [B, 3]
            
        #     # Reshape data back to tile shape
        #     tile_color = tile_color.view(tile_h, tile_w, 3)
        #     # tile_depth = tile_depth.view(tile_h, tile_w, 1)
        #     # acc_alpha = acc_alpha.view(tile_h, tile_w, 1)
            
        #     # Assign to render buffers
        #     self.render_color[h:h+tile_h, w:w+tile_w] = tile_color
        #     # self.render_depth[h:h+tile_h, w:w+tile_w] = tile_depth
        #     # self.render_alpha[h:h+tile_h, w:w+tile_w] = acc_alpha


        # return self.render_color
    
class RasterizationModel_notile(torch.nn.Module):
    def __init__(
        self,
        output_folder, 
        camera_pose,
        fx=2343.0242837919386,
        fy=2343.0242837919386,
        width=2560,
        height=1440,
        camera_type=CameraType.PERSPECTIVE,
        tile_size=16,
    ):
        super().__init__()  # Initialize the base class first

        self.fx = fx 
        self.fy = fy
        self.width = width
        self.height = height
        self.camera_type = camera_type

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_folder = output_folder 
        self.config_path = os.path.join(output_folder, 'config.yml')
        self.load_path = os.path.join(output_folder, './nerfstudio_models/step-000089999.ckpt')
        self.config_path = Path(self.config_path)

        self.config = yaml.load(self.config_path.read_text(), Loader=yaml.Loader)

        metadata={
            'depth_filenames': None, 
            'depth_unit_scale_factor': 0.001, 
            'mask_color': None,
            "env_params":torch.tensor([[1.0,0.0]])
        }

        self.model:SplatfactoEnvModel = self.config.pipeline.model.setup(
            scene_box = SceneBox(
                    aabb=torch.Tensor([
                        [-1., -1., -1.],
                        [ 1.,  1.,  1.]
                    ]),
            ),
            num_train_data = 1440,
            metadata = metadata, 
            device = self.device,
            grad_scaler = None, 
            seed_points = None 
        )
        self.model.training = False
        self.model.to(self.device)

        loaded_state = torch.load(self.load_path, map_location='cuda')
        self.load_state_dict_modify(loaded_state)

        self.transform = np.array([
            [
                0.9996061325073242,
                -0.01975083164870739,
                -0.01993674598634243,
                997.8822021484375
            ],
            [
                -0.01975083164870739,
                0.009563744068145752,
                -0.9997591972351074,
                -42.26317596435547
            ],
            [
                0.01993674598634243,
                0.9997591972351074,
                0.00916987657546997,
                -242.0419158935547
            ]
        ])

        self.scale_factor = 0.0003946526873285077

        camera_pose = self.transform@camera_pose
        camera_pose[:3,3] *= self.scale_factor
        camera_pose = camera_pose[:3,:]
        cam_state = camera_pose

        if cam_state.ndim == 2:
            cam_state = np.expand_dims(cam_state, axis=0)

        self.camera_to_world = torch.FloatTensor( cam_state ).to(self.device)

        self.setup_camera()
        self.prepare_rasterization_coefficients(
            tile_size=tile_size
        )
        # self.shrink_rasterization_coefficients()

        self.prepare_render_coefficients()

    # def shrink_rasterization_coefficients(self):
    #     pass 

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(RasterizationModel_notile, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move custom tensors
            self.sh_coeffs_expanded = self.sh_coeffs_expanded.to(device)
            self.overall_mask = self.overall_mask.to(device)
            self.means_cam_rast = self.means_cam_rast.to(device)
            self.cov2D_rast = self.cov2D_rast.to(device)  
            self.means2D_rast = self.means2D_rast.to(device)
            self.opacities_rast = self.opacities_rast.to(device)
            self.radius = self.radius.to(device)
            
            # Render coefficients
            self.index = self.index.to(device)
            self.T_alpha = self.T_alpha.to(device)
            self.T_alpha_unsorted = self.T_alpha_unsorted.to(device)

            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining

    def load_state_dict_modify(self, loaded_state):

        step = loaded_state['step']
        loaded_state = loaded_state['pipeline']
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)

        state_dict = state 
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}
        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            self.model.load_state_dict(model_state, strict=False)        

    def setup_camera(self):
        self.camera = Cameras(
            camera_to_worlds=self.camera_to_world, 
            fx=self.fx, 
            fy=self.fy,
            cx=self.width/2,
            cy=self.height/2,
            width=self.width,
            height=self.height,
            distortion_params=None,
            camera_type=self.camera_type, 
            metadata=None
        )

        if not isinstance(self.camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # if camera.metadata is None:
        #     camera.metadata = {"env_prams":torch.tensor([0.5,0.5])}

        if self.model.config.sh_degree > 0:
            self.sh_degree_to_use = min(self.model.step // self.model.config.sh_degree_interval, self.model.config.sh_degree)

        if self.model.training:
            # if not camera.shape[0] == 1:
            #     print(">>>>>>>>>>>", camera.shape[0])
            # val = camera.shape[0]
            # string = "Only one camera at a time, received " + str(val)
            # assert camera.shape[0] == 1, string
            optimized_camera_to_world = self.model.camera_optimizer.apply_to_camera(self.camera)
        else:
            optimized_camera_to_world = self.camera.camera_to_worlds


        camera_scale_fac = self.model._get_downscale_factor()
        self.camera.rescale_output_resolution(1 / camera_scale_fac)
        self.viewmat = get_viewmat(optimized_camera_to_world)
        self.W, self.H = int(self.camera.width.item()), int(self.camera.height.item())
        self.model.last_size = (self.H, self.W)
        # camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        self.opacities = self.model.opacities
        self.means = self.model.means
        # base_color = self.model.base_colors
        self.scales = self.model.scales
        self.quats = self.model.quats
        
        # apply the compensation of screen space blurring to gaussians
        self.BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.K = self.camera.get_intrinsics_matrices().cuda()
        if self.model.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.model.config.rasterize_mode)

        if self.model.config.output_depth_during_training or not self.training:
            self.render_mode = "RGB+ED"
        else:
            self.render_mode = "RGB"

        # if camera.metadata is None:
        #     env_params = torch.Tensor([0.5,0.5])
        # else:
        #     env_params = camera.metadata['env_params']
        # env_params_repeat = env_params.repeat(means.shape[0], 1).to(self.device)

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

        view_mats =  get_viewmat(self.camera.camera_to_worlds).to(self.means.device)
        Ks = torch.tensor([[
            [self.camera.fx, 0, self.camera.cx],
            [0, self.camera.fy, self.camera.cy],
            [0, 0, 1]
        ]]).to(self.means.device)
        means = self.means 
        quats = self.quats 
        scales = torch.exp(self.scales)
        opacities = torch.sigmoid(self.opacities)
        width = self.width
        height = self.height
        
        viewmat = view_mats[0]  # [4, 4]
        K = Ks[0]              # [3, 3]

        N = means.size(0)
    
        dtype = means.dtype

        # Step 1: Transform Gaussian centers to camera space
        ones = torch.ones(N, 1, device=self.device, dtype=dtype)
        means_hom = torch.cat([means, ones], dim=1).to(self.device)  # [N, 4]
        means_cam_hom = (viewmat @ means_hom.T).T    # [N, 4]
        means_cam = means_cam_hom[:, :3] / means_cam_hom[:, 3:4]  # [N, 3]

        mask = (means_cam[:,2]>near_plane) & (means_cam[:,2]<far_plane)
        self.overall_mask = mask
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
        scales_matrix = torch.diag_embed(scales).to(self.device)  # [N, 3, 3]
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

        J = torch.zeros(N, 2, 3, device=self.device, dtype=dtype)
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
        self.overall_mask = self.overall_mask & mask 
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
        self.overall_mask = self.overall_mask & mask 
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
        self.overall_mask = self.overall_mask & mask 
        
        # # mask out opacities smaller than threshold
        # threshold, _ = torch.kthvalue(opacities.squeeze(), opacities.shape[0]-300000+1)
        # mask = opacities>=threshold 
        # self.overall_mask = self.overall_mask & mask.squeeze()
        
        means = means[self.overall_mask]
        self.means_cam_rast = means_cam[self.overall_mask]
        self.cov2D_rast = cov2D[self.overall_mask]
        self.means2D_rast = means2D[self.overall_mask]
        self.opacities_rast = opacities[self.overall_mask]
        # colors = colors[self.overall_mask]
        self.radius = radius[self.overall_mask]
        N = means2D.size(0)  # Update N after masking
        
        means3D = means
        c2w = viewmat
        active_sh_degree = 3
        camtoworlds = torch.linalg.inv(c2w)
        rays_o = camtoworlds[:3,3]
        rays_d = means3D - rays_o
        rays_d = rays_d/rays_d.norm(dim=1, keepdim=True)
        sh_coeffs = eval_sh_old(active_sh_degree, rays_d)
        self.sh_coeffs_expanded = sh_coeffs.unsqueeze(2)  # Shape: (424029, 16, 1)

    @torch.no_grad()
    def prepare_render_coefficients(self):
        means2D=self.means2D_rast
        cov2D=self.cov2D_rast
        radii = self.radius
        opacity=self.opacities_rast
        depths = self.means_cam_rast[:,2]
        W=self.width
        H=self.height
        device=self.device
        tile_size=self.tile_size

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
        sorted_opacity = opacity[index]
        self.index = index
        # sorted_color = color[index]
        dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
        
        index_unsorted = torch.empty_like(index)
        range_tensor = torch.arange(len(index), device = index.device)
        index_unsorted[index] = range_tensor
        self.index_unsorted = index_unsorted
        
        gauss_weight = torch.exp(-0.5 * (
            dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
            + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
            + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
            + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))

        gauss_weight_unsorted = gauss_weight[:,index_unsorted]
        alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
        alpha_unsorted = (gauss_weight_unsorted[..., None] * opacity[None]).clip(max=0.99) # B P 1
        T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
        T_unsorted = torch.cat([torch.ones_like(alpha_unsorted[:,:1]), 1-alpha_unsorted[:,:-1]], dim=1).cumprod(dim=1)
        # acc_alpha = (alpha * T).sum(dim=1)
        self.T_alpha = T*alpha 
        self.T_alpha_unsorted = self.T_alpha[:,index_unsorted,:].squeeze()

    # def forward(self, x):
    #     res = self.forward_old(x)
    #     return res 
    
    def forward_old(self, x):
        view_mats =  get_viewmat(self.camera.camera_to_worlds).to(self.means.device)
        Ks = torch.tensor([[
            [self.camera.fx, 0, self.camera.cx],
            [0, self.camera.fy, self.camera.cy],
            [0, 0, 1]
        ]]).to(self.means.device)
        res = rasterize_gaussians_pytorch(
            self.means,
            self.quats/self.quats.norm(dim=-1, keepdim=True),
            torch.exp(self.scales),
            torch.sigmoid(self.opacities),
            x, 
            view_mats,
            Ks,
            self.width,
            self.height
        )
        # pass
        return res  

    def forward(self, x):
        # shs = x[self.overall_mask]
        product = self.sh_coeffs_expanded*x 
        color = torch.sum(product, dim=1)
        # coeffs_T = self.sh_coeffs_expanded.transpose(1, 2)
        # color = torch.bmm(coeffs_T, x) 
        # color = color.squeeze(1)
        color_rgb = torch.nn.functional.relu(color+0.5)
        return color_rgb
        # res2 = (self.T_alpha_unsorted*color_rgb).sum(dim=1)
        res2 = torch.matmul(self.T_alpha_unsorted, color_rgb)
        res2 = res2.reshape(self.H, self.W, -1)
        # return color_rgb
        # color_rgb_sorted = color_rgb[:,self.index]
        # color_rgb_sorted = color_rgb
        # res = (self.T_alpha*color_rgb_sorted).sum(dim=1)
        # res = res.reshape(self.H, self.W, -1)
        
        return res2

        # for tile_data in self.precomputed_tiles:
        #     h = tile_data['h']
        #     w = tile_data['w']
        #     tile_h = tile_data['tile_h']
        #     tile_w = tile_data['tile_w']
        #     T_alpha = tile_data['T_alpha']  # Shape: [B, P]
        #     # acc_alpha = tile_data['acc_alpha']  # Shape: [B]
        #     # tile_depth = tile_data['tile_depth']  # Shape: [B]
        #     sorted_indices = tile_data['sorted_indices']  # Indices into color
            
        #     # Retrieve colors for the Gaussians
        #     sorted_color = color_rgb[sorted_indices]  # Shape: [P, 3]
            
        #     # Compute the tile's color
        #     tile_color = (T_alpha * sorted_color[None]).sum(dim=1)  # Shape: [B, 3]
            
        #     # Reshape data back to tile shape
        #     tile_color = tile_color.view(tile_h, tile_w, 3)
        #     # tile_depth = tile_depth.view(tile_h, tile_w, 1)
        #     # acc_alpha = acc_alpha.view(tile_h, tile_w, 1)
            
        #     # Assign to render buffers
        #     self.render_color[h:h+tile_h, w:w+tile_w] = tile_color
        #     # self.render_depth[h:h+tile_h, w:w+tile_w] = tile_depth
        #     # self.render_alpha[h:h+tile_h, w:w+tile_w] = acc_alpha


        # return self.render_color