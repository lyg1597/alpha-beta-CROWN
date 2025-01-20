import torch 
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np 
from collections import defaultdict

# Define computation as a nn.Module.
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        self.const1 = torch.Tensor([[0,0,0]])
        self.const2 = torch.Tensor([[-1,-1,-1]])

    def forward(self, x: torch.Tensor):
        # Define your computation here.
        res = torch.heaviside(x, values=torch.zeros(x.shape))
        return res 
    
model = MyModel()
x = torch.Tensor(np.array([
    [-1,-2,-3]
]))
res = model(x)
print(res.shape, res)
# Wrap the model with auto_LiRPA.
model_bounded = BoundedModule(model, (x,), device = 'cpu', bound_opts={'conv_mode': 'matrix'})
# Define perturbation. Here we add Linf perturbation to input data.
ptb_x = PerturbationLpNorm(
    norm=np.inf, 
    x_L=torch.Tensor(np.array([
        [-2,-3,-4] 
    ])),
    x_U=torch.Tensor(np.array([
        [0,-1,-2]
    ]))
)
# Make the input a BoundedTensor with the pre-defined perturbation.
inp_x = BoundedTensor(x, ptb_x)
res = model_bounded(inp_x)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model_bounded.compute_bounds(x=(inp_x, ), method = 'alpha-crown')
print('>>>>>> lb, ub with alpha-rown')
print(lb)
print(ub)

