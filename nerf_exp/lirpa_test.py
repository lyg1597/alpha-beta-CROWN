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
from rasterize_model import RasterizationModel, RasterizationModel_notile
from gsplat import rasterization
import cv2 

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

class SplatModel(torch.nn.Module):
    def __init__(
        self,
        output_folder, 
        camera_pose,
        fx=2343.0242837919386,
        fy=2343.0242837919386,
        width=2560,
        height=1440,
        camera_type=CameraType.PERSPECTIVE
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

    @torch.no_grad()
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
        viewmat = get_viewmat(optimized_camera_to_world)
        W, H = int(self.camera.width.item()), int(self.camera.height.item())
        self.model.last_size = (H, W)
        # camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        self.opacities = self.model.opacities
        self.means = self.model.means
        self.ones = torch.ones(self.means.shape[0], 1, device=self.means.device)
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

    def forward(self, x):
        return self._forward_new(x)

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(SplatModel, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move custom tensors
            self.means = self.means.to(device)
            self.ones = self.ones.to(device)
            self.quats = self.quats.to(device)
            self.scales = self.scales.to(device)

            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining
    
    def _forward_new(self, x):
        # for i in range(x.shape[0]):
            # metadata = {"env_params":x[i]}
            # camera.to(self.device)
            # tmp = self.model.get_outputs(camera)
            # img = tmp['rgb']
            # res_list.append(img)
        # env_params = x
        # print(env_params.device)
        # env_params_repeat = env_params.repeat(self.means.shape[0], 1)
        env_params_repeat = torch.matmul(self.ones, x)
        # return env_params_repeat
        # print(env_params.device)
        # colors = self.model.color_nn(
        #     tmp_means,
        #     tmp_quats,
        #     tmp_scales,
        #     env_params
        # )
        colors = self.model.color_nn(
            self.means,
            self.quats,
            self.scales,
            env_params_repeat
        )

        # res_list.append(colors)

        # res_tensor = torch.stack(res_list, dim=0).view(x.shape[0], -1)  # Shape: (N, 480, 640, 3)
        return colors


    def _forward_old(self, x):
        res_list = []
        for i in range(x.shape[0]):
            metadata = {"env_params":x[i]}
            camera = Cameras(
                camera_to_worlds=self.camera_to_world, 
                fx=self.fx, 
                fy=self.fy,
                cx=self.width/2,
                cy=self.height/2,
                width=self.width,
                height=self.height,
                distortion_params=None,
                camera_type=self.camera_type, 
                metadata=metadata
            )
            camera.to(self.device)
            tmp = self.model.get_outputs(camera)
            img = tmp['rgb']
            res_list.append(img)

        res_tensor = torch.stack(res_list, dim=0).view(x.shape[0], -1)  # Shape: (N, 480, 640, 3)

        # metadata={"env_params":x} 
        # cameras = Cameras(
        #     camera_to_worlds=self.camera_to_world, 
        #     fx=self.fx, 
        #     fy=self.fy,
        #     cx=self.width/2,
        #     cy=self.height/2,
        #     width=self.width,
        #     height=self.height,
        #     distortion_params=None,
        #     camera_type=self.camera_type, 
        #     metadata=metadata
        # )
        # cameras.to(self.device)
        # res_list = []
        # for i in range(len(cameras)):
        #     camera = cameras[i:i+1]
        #     tmp = self.model.get_outputs(camera)
        #     img = tmp['rgb']
        #     res_list.append(img)

        # res_tensor = torch.stack(res_list, dim=0).view(len(cameras), -1)  # Shape: (N, 480, 640, 3)

        return res_tensor

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
    f = 2343.0242837919386

    model = SplatModel(
        output_folder=output_folder,
        camera_pose = camera_pose,
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

    with torch.no_grad():
        res = rasterization(
            model.means,
            model.quats/ model.quats.norm(dim=-1, keepdim=True),
            torch.exp(model.scales),
            torch.sigmoid(model.opacities).squeeze(-1),
            res_2d,
            view_mats,      # 
            Ks,
            model.width,
            model.height,
            sh_degree=3
        )

    rendered = res[0]
    print(rendered.shape)
    rendered = rendered.detach().cpu().numpy()[0]
    print(rendered.shape)
    image_uint8 = (np.clip(rendered, 0, 1)*255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fn, image_bgr)
    plt.imshow(rendered)
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

    model = RasterizationModel(
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
    
    # my_input = torch.clone(res_2d[model.overall_mask])
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