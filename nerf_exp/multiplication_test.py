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
    [-2,0,2]
]))
y = torch.Tensor(np.array([
    [-0.5,-0.5,-0.5]
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
        [-3,-1,-1] 
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
lb, ub = model_bounded.compute_bounds(x=(inp_y, inp_z), method = 'ibp')
print('>>>>>> lb, ub with ibp')
print(lb)
print(ub)

print('>>>>>> lb, ub with crown')
required_A = defaultdict(set)
required_A[model_bounded.output_name[0]].add(model_bounded.input_name[0])
lb, ub, A = model_bounded.compute_bounds(x=(inp_y, inp_z), method = 'crown', return_A=True, needed_A_dict=required_A)
lA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lA']
uA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['uA']
lbias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lbias']
ubias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['ubias']
print(lA)
print(uA)
print(lbias)
print(ubias)
lb, ub = model_bounded.compute_bounds(x=(inp_y, inp_z), method = 'crown')
print(lb)
print(ub)

print('>>>>>> lb, ub with alpha-crown')
required_A = defaultdict(set)
required_A[model_bounded.output_name[0]].add(model_bounded.input_name[0])
lb, ub, A = model_bounded.compute_bounds(x=(inp_y, inp_z), method = 'alpha-crown', return_A=True, needed_A_dict=required_A)
lA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lA']
uA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['uA']
lbias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lbias']
ubias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['ubias']
print(lA)
print(uA)
print(lbias)
print(ubias)
lb, ub = model_bounded.compute_bounds(x=(inp_y, inp_z), method = 'alpha-crown')
print(lb)
print(ub)
