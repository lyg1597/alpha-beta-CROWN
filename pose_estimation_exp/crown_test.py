from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from gatenet2 import GateNet2 
import torch 
import numpy as np 

def load_model(checkpoint_path, device):
    """Loads the trained GateNet model from a checkpoint."""
    config = {
        'input_shape': (3, 50, 50),
        'output_shape': (6,),
        'batch_norm_decay': 0.99,
        'batch_norm_epsilon': 1e-3
    }
    model = GateNet2(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    checkpoint_path = '././checkpoints/checkpoint_epoch_15.pth'

    model = load_model(checkpoint_path, 'cuda')

    data = np.load('dozer2_data_0.001.npz')
    images_lb = data['images_lb']
    images_ub = data['images_ub']
    camera_poses = data['images_noptb']
    image_concrete = data['images_concrete']

    image_lb = torch.Tensor(images_lb[0]).permute(2,0,1)[None].to('cuda')
    image_ub = torch.Tensor(images_ub[0]).permute(2,0,1)[None].to('cuda')
    image_concrete = torch.Tensor(image_concrete).permute(2,0,1)[None].to('cuda')
    image_lb, image_ub = torch.minimum(image_lb, image_ub), torch.maximum(image_lb, image_ub)
    print((image_ub-image_lb).max())
    image_center = (image_lb+image_ub)/2
    print(model(image_concrete), camera_poses)

    model_bounded = BoundedModule(model, image_center)

    ptb = PerturbationLpNorm(norm=np.inf, x_L = image_lb, x_U = image_ub)
    image_bounded = BoundedTensor(image_center, ptb)
    prediction = model(image_bounded)
    lb, ub = model_bounded.compute_bounds(x=(image_bounded,), method="alpha-crown")
    print(lb, ub, camera_poses)