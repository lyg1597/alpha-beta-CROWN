from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
# from pathlib import Path 
# import yaml 
# from splatfactoenv.splatfactoenv_model import SplatfactoEnvModel
# from nerfstudio.data.scene_box import SceneBox
import numpy as np 
# from nerfstudio.cameras.cameras import Cameras, CameraType
import matplotlib.pyplot as plt 
from rasterize_model import RasterizationModel, RasterizationModel_notile
from gsplat import rasterization
# import cv2 
from rasterization_pytorch import rasterize_gaussians_pytorch
from splat_model import SplatModel

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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/gazebo5_transformed_env-3/splatfacto-env/2024-10-18_161243/')

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

    fn = "frames_00775_gs.png"

    width=2560
    height=1440
    f = 2300.0

    model = SplatModel(
        output_folder=output_folder,
        camera_pose = camera_pose,
        checkpoint="step-000089999.ckpt",
        width=width,
        height=height,
        fx = f,
        fy = f,
    )
    my_input = torch.Tensor(np.array([[0.0,0.0]])).to(torch.device('cuda'))
    with torch.no_grad():
        res_2d = model(my_input)
    print(res_2d.shape)

    # print(">>>>>> Starting Bounded Module")
    # model_bounded = BoundedModule(model, my_input, device=res_2d.device)
    # print(">>>>>> Starting PerturbationLpNorm")
    # ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # print(">>>>>> Starting BoundedTensor")
    # my_input = BoundedTensor(my_input, ptb)
    # prediction = model_bounded(my_input)
    # lb, ub = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    # print(">>>>>> Done")
    # print(lb.shape)
    # print(ub.shape)


    # view_mats = torch.cat((model.camera.camera_to_worlds, addon), dim=1)
    view_mats = get_viewmat(model.camera.camera_to_worlds)
    Ks = torch.tensor([[
        [model.camera.fx, 0, model.camera.cx],
        [0, model.camera.fy, model.camera.cy],
        [0, 0, 1]
    ]]).to(torch.device('cuda'))

    # with torch.no_grad():
    #     res = rasterization(
    #         model.means,
    #         model.quats/ model.quats.norm(dim=-1, keepdim=True),
    #         torch.exp(model.scales),
    #         torch.sigmoid(model.opacities).squeeze(-1),
    #         res_2d,
    #         view_mats,      # 
    #         Ks,
    #         model.width,
    #         model.height,
    #         sh_degree=3
    #     )

    # rendered = res[0]
    # print(rendered.shape)
    # rendered = rendered.detach().cpu().numpy()[0]
    # print(rendered.shape)
    # image_uint8 = (np.clip(rendered, 0, 1)*255).astype(np.uint8)
    # image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(fn, image_bgr)
    # plt.imshow(rendered)
    # plt.show()

    with torch.no_grad():
        res = rasterize_gaussians_pytorch(
            model.means, 
            model.quats/ model.quats.norm(dim=-1, keepdim=True),
            torch.exp(model.scales),
            torch.sigmoid(model.opacities).squeeze(-1),
            res_2d,
            view_mats, 
            Ks,
            model.width,
            model.height
        )
    print(res.shape)
    res = res.detach().cpu().numpy()
    plt.imshow(res)
    plt.show()

    # with torch.no_grad():
    #     res, colors_gt, rasterizer_input = model.model.get_outputs(
    #         model.camera,
    #         debug = True
    #     )
    # rendered = res['rgb']
    # print(rendered.shape)
    # rendered = rendered.detach().cpu().numpy()
    # plt.imshow(rendered)
    # plt.show()

    # model_bounded = BoundedModule(model, my_input)
    # ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # my_input = BoundedTensor(my_input, ptb)
    # prediction = model_bounded(my_input)
    # lb, ub = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    # print(lb,ub)

    model = RasterizationModel_notile(
        output_folder=output_folder,
        camera_pose = camera_pose,
        width=width,
        height=height,
        fx = f,
        fy = f,
        tile_size=8
    )
    model.to(torch.device('cuda'))
    # my_input = torch.rand((602465, 16, 3)).to(torch.device('cuda'))
    # my_input[:,0,:] = 0
    tmp = res_2d[model.overall_mask]
    print(tmp.shape)
    with torch.no_grad():
        res = model(tmp)
    print(res.shape)
    res = res.detach().cpu().numpy()
    plt.imshow(res)
    plt.show()
    
    my_input = torch.clone(res_2d[model.overall_mask])
    print(">>>>>> Starting Bounded Module")
    model_bounded = BoundedModule(model, my_input, device=res_2d.device)
    print(">>>>>> Starting PerturbationLpNorm")
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    print(">>>>>> Starting BoundedTensor")
    my_input = BoundedTensor(my_input, ptb)
    prediction = model_bounded(my_input)
    lb, ub = model_bounded.compute_bounds(x=(my_input, ), method='backward')
    print(">>>>>> Done")
    print(lb.shape)
    print(ub.shape)

    # res_lb = torch.matmul(model.T_alpha_unsorted, lb).reshape(model.H, model.W, -1)
    # res_ub = torch.matmul(model.T_alpha_unsorted, ub).reshape(model.H, model.W, -1)


    # plt.figure(0)
    # res_lb = res_lb.detach().cpu().numpy()
    # plt.imshow(res_lb)

    # plt.figure(1)
    # res_ub = res_ub.detach().cpu().numpy()
    # plt.imshow(res_ub)

    # plt.show()

    # plt.show()