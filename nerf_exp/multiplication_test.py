import torch 
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np 
from collections import defaultdict

# Define computation as a nn.Module.
class MyModel(torch.nn.Module):
    def forward(self, x, y):
        # Define your computation here.
        res = x*y 
        return res 
    
model = MyModel()
x = torch.Tensor(np.array([
    [-2, 0, 2]
]))
y = torch.Tensor(np.array([
    [0,0,0]
]))
z = torch.Tensor(
    np.array([
        [10, 10.5, 11]
    ])
)
# Wrap the model with auto_LiRPA.
model_bounded = BoundedModule(model, (x,y), device = 'cpu', bound_opts={'conv_mode': 'matrix'})
# Define perturbation. Here we add Linf perturbation to input data.
ptb_x = PerturbationLpNorm(
    norm=np.inf, 
    x_L=torch.Tensor(np.array([
        [-3,-1,1] 
    ])),
    x_U=torch.Tensor(np.array([
        [-1,1,3]
    ]))
)
ptb_y = PerturbationLpNorm(
    norm=np.inf, 
    x_L=torch.Tensor(np.array([
        [-1,-1,-1] 
    ])),
    x_U=torch.Tensor(np.array([
        [1,1,1]
    ]))
)
ptb_z = PerturbationLpNorm(
    norm=np.inf, 
    x_L=torch.Tensor(np.array([
        [9, 9.5, 10] 
    ])),
    x_U=torch.Tensor(np.array([
        [11, 11.5, 12]
    ]))
)
# Make the input a BoundedTensor with the pre-defined perturbation.
inp_x = BoundedTensor(x, ptb_x)
inp_y = BoundedTensor(y, ptb_y)
inp_z = BoundedTensor(z, ptb_z)
res = model_bounded(inp_x, inp_y)
model_bounded.visualize('mult')
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model_bounded.compute_bounds(x=(inp_x, inp_y), method = 'ibp')
print('>>>>>> lb, ub with ibp')
print(lb)
print(ub)

lb, ub = model_bounded.compute_bounds(x=(inp_x, inp_y), method = 'crown')
print('>>>>>> lb, ub with crown')
print(lb)
print(ub)

lb, ub = model_bounded.compute_bounds(x=(inp_x, inp_y), method = 'alpha-crown')
print('>>>>>> lb, ub with alpha-crown')
print(lb)
print(ub)
