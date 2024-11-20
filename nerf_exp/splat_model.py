from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import os 
from pathlib import Path 
import yaml 
from splatfactoenv.splatfactoenv_model import SplatfactoEnvModel
from nerfstudio.data.scene_box import SceneBox
import numpy as np 
from nerfstudio.cameras.cameras import Cameras, CameraType

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
        checkpoint,
        camera_pose,
        fx=2343.0242837919386,
        fy=2343.0242837919386,
        width=2560,
        height=1440,
        camera_type=CameraType.PERSPECTIVE,
        use_sh = True
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
        self.load_path = os.path.join(output_folder, f'./nerfstudio_models/{checkpoint}')
        self.config_path = Path(self.config_path)

        self.config = yaml.load(self.config_path.read_text(), Loader=yaml.Loader)
        self.use_sh = use_sh

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

        # if self.model.training:
        #     # if not camera.shape[0] == 1:
        #     #     print(">>>>>>>>>>>", camera.shape[0])
        #     # val = camera.shape[0]
        #     # string = "Only one camera at a time, received " + str(val)
        #     # assert camera.shape[0] == 1, string
        #     optimized_camera_to_world = self.model.camera_optimizer.apply_to_camera(self.camera)
        # else:
        optimized_camera_to_world = self.camera_to_world


        camera_scale_fac = self.model._get_downscale_factor()
        self.camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        W, H = int(self.camera.width.item()), int(self.camera.height.item())
        self.model.last_size = (H, W)
        # camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        self.opacities = self.model.opacities
        self.means = self.model.means
        # base_color = self.model.base_colors
        self.scales = self.model.scales
        self.quats = self.model.quats

        # self.opacities = self.opacities.unsqueeze(0)
        # self.means = self.means.unsqueeze(0)
        # self.scales = self.scales.unsqueeze(0)
        # self.quats = self.quats.unsqueeze(0)
        
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
        env_params_repeat = x.repeat(self.means.shape[0], 1)
        # env_params_repeat = torch.matmul(self.ones, x)
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