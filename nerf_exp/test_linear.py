from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import numpy as np 

# Define computation as a nn.Module.
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.Tensor(np.array([
            [1,0],
            [0,1]
        ]))
        self.b = torch.Tensor(np.array([
            [2],
            [3]
        ]))

    def forward(self, x):
        # Define your computation here.
        return self.W@x+self.b

model = MyModel()
my_input = torch.Tensor(np.array([
    [3],
    [4]
]))
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input)
model.visualize('test_linear')
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(my_input,), method="backward")
print(lb, ub)