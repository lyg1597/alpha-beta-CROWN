import torch 
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np 
from collections import defaultdict

# Define computation as a nn.Module.
class MyModel(torch.nn.Module):
    def forward(self, x):
        # Define your computation here.
        res = torch.inverse(x)
        return res 
    
model = MyModel()
x = torch.Tensor(np.array([
    [2,-1],
    [-1,2]
]))
model_bounded = BoundedModule(model, (x), device = 'cpu', bound_opts={'conv_mode': 'matrix'})
# Define perturbation. Here we add Linf perturbation to input data.
ptb_x = PerturbationLpNorm(
    norm=np.inf, 
    eps=0.1
)
# Make the input a BoundedTensor with the pre-defined perturbation.
inp_x = BoundedTensor(x, ptb_x)
res = model_bounded(inp_x)
model_bounded.visualize('inv')
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model_bounded.compute_bounds(x=(inp_x,), method = 'ibp')
print('>>>>>> lb, ub with ibp')
print(lb)
print(ub)

print('>>>>>> lb, ub with crown')
required_A = defaultdict(set)
required_A[model_bounded.output_name[0]].add(model_bounded.input_name[0])
lb, ub, A = model_bounded.compute_bounds(x=(inp_x,), method = 'crown', return_A=True, needed_A_dict=required_A)
lA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lA']
uA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['uA']
lbias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lbias']
ubias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['ubias']
print(lA)
print(uA)
print(lbias)
print(ubias)
lb, ub = model_bounded.compute_bounds(x=(inp_x,), method = 'crown')
print(lb)
print(ub)

print('>>>>>> lb, ub with alpha-crown')
required_A = defaultdict(set)
required_A[model_bounded.output_name[0]].add(model_bounded.input_name[0])
lb, ub, A = model_bounded.compute_bounds(x=(inp_x,), method = 'alpha-crown', return_A=True, needed_A_dict=required_A)
lA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lA']
uA: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['uA']
lbias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['lbias']
ubias: torch.Tensor = A[model_bounded.output_name[0]][model_bounded.input_name[0]]['ubias']
print(lA)
print(uA)
print(lbias)
print(ubias)
lb, ub = model_bounded.compute_bounds(x=(inp_x,), method = 'alpha-crown')
print(lb)
print(ub)
