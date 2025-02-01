import torch 
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np 
from collections import defaultdict

def linear_bounds(A,b,x_L, x_U):
    pos_mask = (A>=0).float() 
    neg_mask = 1.0-pos_mask 

    A_pos = A*pos_mask 
    A_neg = A*neg_mask 

    fmin = torch.einsum('iabc,ic->iab',A_pos,x_L)+torch.einsum('iabc,ic->iab',A_neg,x_U)+b 
    fmax = torch.einsum('iabc,ic->iab',A_pos,x_U)+torch.einsum('iabc,ic->iab',A_neg,x_L)+b
    return fmin, fmax

# Define computation as a nn.Module.
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        self.const1 = torch.Tensor([[0,0,0]])
        self.const2 = torch.Tensor([[-1,-1,-1]])

    def forward(self, x: torch.Tensor):
        # Define your computation here.
        res = torch.sum(x, dim=1)
        return res 
    
model = MyModel()
x = torch.Tensor(np.array([
    [1,2,3],
    [1,2,3],
    [1,2,3],
]))
res = model(x)
print(res.shape, res)
# Wrap the model with auto_LiRPA.
model_bounded = BoundedModule(model, (x,), device = 'cpu', bound_opts={'conv_mode': 'matrix'})
# Define perturbation. Here we add Linf perturbation to input data.
ptb_x = PerturbationLpNorm(
    norm=np.inf, 
    x_L=torch.Tensor(np.array([
        [1.0,3.4,1.5],
        [1.0,3.4,1.5],
        [1.0,3.4,1.5],
    ])),
    x_U=torch.Tensor(np.array([
        [1.0,3.4,1.5], 
        [1.0,3.4,1.5], 
        [1.0,3.4,1.5], 
    ]))
)
xl = ptb_x.x_L
xu = ptb_x.x_U 
# Make the input a BoundedTensor with the pre-defined perturbation.
inp_x = BoundedTensor(x, ptb_x)
res = model_bounded(inp_x)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model_bounded.compute_bounds(x=(inp_x, ), method = 'crown')

Al = (torch.log(xu)-torch.log(xl))/(xu-xl) 
bl = torch.log(xl)-(torch.log(xu)-torch.log(xl))/(xu-xl)*xl

Au = 1/((xu+xl)/2)
bu = torch.log((xl+xu)/2)-1/((xu+xl)/2)*((xu+xl)/2)

# diffL_part, _ = linear_bounds(Al, bl, xl, xu)    # 1*N*BS
# _, diffU_part = linear_bounds(Au, bu, xl, xu)    # 1*N*BS

print('>>>>>> lb, ub with alpha-rown')
print(lb)
print(ub)

print(Al*xl+bl)
print(Au*xu+bu)

