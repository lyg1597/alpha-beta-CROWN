import torch 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from rasterization_pytorch import rasterize_gaussians_pytorch_rgb
from scipy.spatial.transform import Rotation 

width=50
height=50
f = 70

dt = {
    "transform": [
        [
            1.0,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0
        ]
    ],
    "scale": 1.0
}

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

def sample_point():
    pnt = np.random.uniform([-5.0,-5.0,-0.4], [5.0,5.0,5.0])
    if -0.65<=pnt[0]<=0.65 and -1.2<=pnt[1]<=1.2 and -0.4<=pnt[2]<=1.1:
        pnt = np.random.uniform([-5.0,-5.0,-0.4], [4.0,4.0,1.6])
    # pnt = np.array([3.7, 0, 0.8])
    return pnt 

if __name__ == "__main__":
    num_samples = 20000
    os.mkdir(f'./lego_dataset_{num_samples}')
    os.mkdir(f'./lego_dataset_{num_samples}/images')
    os.mkdir(f'./lego_dataset_{num_samples}/poses')

    transform = np.array(dt['transform'])
    transform_ap = np.vstack((transform, np.array([0,0,0,1])))
    scale = dt['scale']

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/dozer2/splatfacto/2025-03-11_161130')
    checkpoint = "step-000029999.ckpt"
    
    checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
    res = torch.load(checkpoint_fn)
    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']
    colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

    mask_opacities = torch.sigmoid(opacities).squeeze()>0.15
    mask_scale = torch.all(scales>-8.0, dim=1)
    means = means[mask_scale&mask_opacities]
    quats = quats[mask_scale&mask_opacities]
    opacities = opacities[mask_scale&mask_opacities]
    scales = scales[mask_scale&mask_opacities]
    colors = colors[mask_scale&mask_opacities]

    # means_tmp = torch.cat((means, torch.ones(means.shape[0],1).to('cuda')), dim=1)
    # means_w = torch.inverse(torch.Tensor(transform_ap).to('cuda'))@means_tmp.transpose(0,1)/scale

    for i in range(num_samples):
        print(i)
        pnt = sample_point()    
        yaw = np.arctan2(pnt[1], pnt[0])
        pitch = np.pi/2-np.arctan2(pnt[2], np.sqrt(pnt[1]**2+pnt[0]**2))
        # ori = i*np.pi*2/200
        new_ori = np.array([pitch, 0, yaw])
        new_ori_mat = Rotation.from_euler('xyz',new_ori).as_matrix()
        new_pos = np.array([pnt[1],-pnt[0],pnt[2]])
        new_mat = np.zeros((4,4))
        new_mat[:3,:3] = new_ori_mat 
        new_mat[:3,3] = new_pos 
        camera_pose = new_mat

        # transform = np.array(dt['transform'])
        # scale = dt['scale']
        tmp = np.linalg.inv(np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]))
        camera_pose_transformed = tmp@transform_ap@camera_pose
        camera_pose_transformed = camera_pose_transformed[:3,:]
        camera_pose_transformed[:3,3] *= scale 
        camera_pose_transformed = torch.Tensor(camera_pose_transformed)[None].to(means.device)

        # camera_to_worlds = torch.Tensor(camera_pose)[None].to(means.device)
        camera_to_world = torch.Tensor(camera_pose_transformed)[None].to(means.device)

        view_mats = get_viewmat(camera_pose_transformed)
        Ks = torch.tensor([[
            [f, 0, width/2],
            [0, f, height/2],
            [0, 0, 1]
        ]]).to(torch.device('cuda'))

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
                height,
                eps2d=0.0
            )
        res_rgb: torch.Tensor = res['render']
        # print(res_rgb.shape)
        res_rgb = res_rgb[:,...,:3]
        res_rgb = res_rgb.detach().cpu().numpy()
        res_rgb = res_rgb.clip(min=0.0, max=1.0)
        plt.imsave(f"./lego_dataset_{num_samples}/images/image_{i}.png", res_rgb)
        with open(f"./lego_dataset_{num_samples}/poses/image_{i}.txt", "w+") as fp:
            fp.write(f"{new_pos[0]} {new_pos[1]} {new_pos[2]} {new_ori[0]} {new_ori[1]} {new_ori[2]}")
        # plt.figure(0)
        # plt.imshow(res_rgb)
        # plt.show()
