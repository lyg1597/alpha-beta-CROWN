import torch 
import numpy as np 
import os 
from splat_model import SplatModel
from scipy.spatial.transform import Rotation 

def get_viewmat(camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = camera_to_world[:, :3, :3]  # 3 x 3
    T = camera_to_world[:, :3, 3:4]  # 3 x 1
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

@torch.no_grad()
def strip_gaussians(
    camera_to_world,    # Camera info 
    tile_tl,            # top left pixel of tile
    tile_br,            # bottom right pixel of tile 
    means,
    quats,
    scales,
    K, 
    eps: torch.FloatTensor,
    near_plane=0.01, 
    far_plane=1e10, 
    eps2d=0.3,
    radius_clip=0.0,
    overall_mask=None
):
    '''
    Using montecarlo simulation to get a rough estimation of what gaussians
    should be considered for the given camera pose range
    TODO: Make this a concrete overapproximation instead of montecarlo 
    estimation
    '''

    means = self.means 
    quats = self.quats 
    scales = torch.exp(self.scales)
    width = self.width
    height = self.height

    K = self.K[0]

    N = means.size(0)

    dtype = means.dtype

    camera_to_world = self.camera_to_world
    camera_poses = self.sample_cameras(camera_to_world, eps, 100)
    if overall_mask is None:
        overall_mask = torch.empty((camera_poses.shape[0], means.shape[0]), dtype=torch.bool).to(self.device) 
        overall_mask.fill_(True)
        for i in range(camera_poses.shape[0]):
            view_mats =  get_viewmat(camera_poses[i]).to(self.means.device)
            viewmat = view_mats[0]
            # Step 1: Transform Gaussian centers to camera space
            ones = torch.ones(N, 1, device=self.device, dtype=dtype)
            means_hom = torch.cat([means, ones], dim=1).to(self.device)  # [N, 4]
            means_cam_hom = (viewmat @ means_hom.T).T    # [N, 4]
            means_cam = means_cam_hom[:, :3] / means_cam_hom[:, 3:4]  # [N, 3]

            mask = (means_cam[:,2]>near_plane) & (means_cam[:,2]<far_plane)
            overall_mask[i,:] = overall_mask[i,:] & mask

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
            cov2D[:,0,0] = cov2D[:,0,0]+eps2d 
            cov2D[:,1,1] = cov2D[:,1,1]+eps2d 
            det_blur = cov2D[:,0,0]*cov2D[:,1,1]-cov2D[:,0,1]*cov2D[:,1,0]
            det = det_blur

            mask = det_blur>0
            overall_mask[i,:] = overall_mask[i,:]&mask 
            
            # Step 7: Check if points are in image region 
            # Take 3 sigma as the radius 
            b = 0.5*(cov2D[:,0,0]+cov2D[:,1,1])
            v1 = b+torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
            v2 = b-torch.sqrt(torch.max(torch.ones(det.shape).to(det.device)*0.1, b*b-det))
            radius = torch.ceil(3*torch.sqrt(torch.max(v1, v2)))
            
            mask = radius>radius_clip
            overall_mask[i,:] = overall_mask[i,:]&mask 

            mask = (means2D[:,0]+radius>0) & (means2D[:,0]-radius<width) & (means2D[:,1]+radius>0) & (means2D[:,1]-radius<height)
            overall_mask[i,:] = overall_mask[i,:]&mask 
        overall_mask = torch.any(overall_mask, dim=0)
    self.means = torch.nn.Parameter(self.means[overall_mask,:])
    self.quats = torch.nn.Parameter(self.quats[overall_mask,:])
    self.scales = torch.nn.Parameter(self.scales[overall_mask,:])
    self.opacities = torch.nn.Parameter(self.opacities[overall_mask,:])
    self.opacities_rast = self.opacities_rast[:,:,overall_mask, :]
    self.cov_world = self.cov_world[:, overall_mask, :, :]
    self.J = self.J[:, overall_mask,:,:]
    self.means_hom_tmp = self.means_hom_tmp[:,:,overall_mask]
    self.overall_mask = overall_mask

dt = {
    "transform": [
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
    ],
    "scale": 0.0003946526873285077
}

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/gazebo5_transformed_env-1/splatfacto-env-rgb/2024-11-18_154538/')
    checkpoint = "step-000029999.ckpt"

    model_file = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    res = torch.load(model_file)
    
    camera_pose = np.array([
        [
            -0.23762398510466104,
            0.44276476982071006,
            -0.864577469234882,
            -2230.7194253135594
        ],
        [
            -2.9884813341042206e-16,
            0.8900715974578106,
            0.45582074480973456,
            358.8874872340502
        ],
        [
            0.9713572163231092,
            0.10831394187506413,
            -0.21150236001639652,
            -166.52500219585227
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])
    transform = np.array(dt['transform'])
    scale = dt['scale']
    camera_pose_transformed = transform@camera_pose
    camera_pose_transformed[:3,3] *= scale 
    camera_pose_transformed = get_viewmat(camera_pose_transformed)

    camera_pos = camera_pose_transformed[:,:3]
    camera_ori = Rotation.from_matrix(camera_pose_transformed[:3,:3]).as_euler('xyz')
    camera_pose_transformed = [
        camera_ori[0], 
        camera_ori[1], 
        camera_ori[2], 
        camera_pos[0], 
        camera_pos[1], 
        camera_pos[2]
    ]
    fn = "frames_00775_gs.png"

    width=16
    height=16
    f = 1200.0

    model = SplatModel(
        output_folder=output_folder,
        camera_pose = camera_pose,
        checkpoint=checkpoint,
        width=width,
        height=height,
        fx = f,
        fy = f,
        use_sh = False
    )

    my_input = torch.Tensor(np.array([[0,0]])).to(torch.device('cuda'))
    with torch.no_grad():
        res_2d = model(my_input)
    print(res_2d.shape)

    means = model.means 
    quats = model.quats 
    opacities = model.opacities 
    colors = res_2d[:,:3]


