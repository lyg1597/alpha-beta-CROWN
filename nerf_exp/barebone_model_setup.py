from splatfactoenv.splatfactoenv_model import SplatfactoEnvModel
import yaml 
import os 
from pathlib import Path

import numpy as np 
import torch 

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox

import matplotlib.pyplot as plt 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

script_dir = os.path.dirname(os.path.realpath(__file__))

output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/gazebo5_transformed_env-3/splatfacto-env/2024-10-18_161243/')
config_path = os.path.join(output_folder, 'config.yml')
load_path = os.path.join(output_folder, './nerfstudio_models/step-000089999.ckpt')
config_path = Path(config_path)

config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
from nerfstudio.utils.eval_utils import eval_setup

metadata={
    'depth_filenames': None, 
    'depth_unit_scale_factor': 0.001, 
    'mask_color': None,
    "env_params":torch.tensor([[1.0,0.0]])
}

model:SplatfactoEnvModel = config.pipeline.model.setup(
    scene_box = SceneBox(
            aabb=torch.Tensor([
                [-1., -1., -1.],
                [ 1.,  1.,  1.]
            ]),
    ),
    num_train_data = 1440,
    metadata = metadata, 
    device = device,
    grad_scaler = None, 
    seed_points = None 
)
model.to(device)
loaded_state = torch.load(load_path, map_location='cuda')

step = loaded_state['step']
loaded_state = loaded_state['pipeline']
state = {
    (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
}
model.update_to_step(step)

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
    model.load_state_dict(model_state, strict=True)
except RuntimeError:
    model.load_state_dict(model_state, strict=False)

# _, pipeline, _, step = eval_setup(
#     config_path,
#     eval_num_rays_per_chunk=None,
#     test_mode='inference'
# )

# model2 = pipeline.model 

# assert model == model2 

camera_pose = np.array([
    [
        0.07585322596629693,
        0.2607750091672529,
        -0.9624150262253418,
        -3050.83116632528
    ],
    [
        -2.7412222166117605e-16,
        0.9651957610452591,
        0.26152847428198583,
        407.8386916124485
    ],
    [
        0.9971189939573442,
        -0.019837778456332084,
        0.07321321216427805,
        114.17181528627293
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0
    ]
])

transform = np.array([
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

scale_factor = 0.0003946526873285077

camera_pose = transform@camera_pose
camera_pose[:3,3] *= scale_factor
camera_pose = camera_pose[:3,:]
cam_state = camera_pose

if cam_state.ndim == 2:
    cam_state = np.expand_dims(cam_state, axis=0)

camera_to_world = torch.FloatTensor( cam_state )

metadata={"env_params":torch.tensor([1.0,0.0])}

camera = Cameras(
    camera_to_worlds=camera_to_world, 
    fx=2343.0242837919386, 
    fy=2343.0242837919386,
    cx=640/2,
    cy=480/2,
    width=640,
    height=480,
    distortion_params=None,
    camera_type=CameraType.PERSPECTIVE, 
    metadata=metadata
)

camera = camera.to(device)

for i in range(10):
    with torch.no_grad():
        tmp = model.get_outputs(camera)
    img = tmp['rgb'].cpu().detach().numpy()
    print(img.shape)

plt.imshow(img)
plt.show()