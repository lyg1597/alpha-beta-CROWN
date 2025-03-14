import torch 
import os 

script_dir = os.path.dirname(os.path.realpath(__file__))
output_folder = os.path.join(script_dir, '../../nerfstudio/outputs/chair2/splatfacto/2025-03-13_233216')
checkpoint = "step-000029999.ckpt"

checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', checkpoint)
res = torch.load(checkpoint_fn)
means = res['pipeline']['_model.gauss_params.means']
quats = res['pipeline']['_model.gauss_params.quats']
opacities = res['pipeline']['_model.gauss_params.opacities']
scales = res['pipeline']['_model.gauss_params.scales']
colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])

means[:,1] += 0.15
res['pipeline']['_model.gauss_params.means'] = means

checkpoint_fn = os.path.join(output_folder, 'nerfstudio_models', "step-000029999_modified.ckpt")
torch.save(res, checkpoint_fn)