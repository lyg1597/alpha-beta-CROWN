from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np 
# from torch import nn
import torch 

# Define computation as a nn.Module.
class MyModel(torch.nn.Module):
    def __init__(self, N=10):
        super().__init__()
        self.N = N
        self.reshape = torch.ones((1,self.N,3,3))
        self.weight = torch.rand((1,self.N,3,3)) # 1*N*3*3

    def forward(self, x):
        # Define your computation here.
        tmp = torch.matmul(x, self.weight)  
        res = torch.matmul(tmp, x.transpose(-1, -2))
        return res

    # def forward(self, x):
    #     # Define your computation here.
    #     x = x*self.reshape
    #     res = x@self.weight@x.transpose(-1,-2)
    #     return res

model = MyModel()
my_input = torch.rand((1,1,3,3)) # 1*1*3*3
res = model(my_input)
print(res.shape)
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(my_input,), method="IBP")
print(lb, ub)
