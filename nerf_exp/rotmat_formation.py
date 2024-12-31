from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch 
import numpy as np 

# Define computation as a nn.Module.
class MyModel(torch.nn.Module):
    def forward(self, x):
        # Define your computation here.
        gamma = x[:,0:1]
        beta = x[:,1:2]
        alpha = x[:,2:3]
        R00 = torch.cos(beta)*torch.cos(gamma)
        R01 = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma)-torch.cos(alpha)*torch.sin(gamma)
        R02 = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma)+torch.sin(alpha)*torch.sin(gamma)
        R03 = x[:,3:4]
        R10 = torch.cos(beta)*torch.sin(gamma)
        R11 = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma)+torch.cos(alpha)*torch.cos(gamma)
        R12 = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma)-torch.sin(alpha)*torch.cos(gamma)
        R13 = x[:,4:5]
        R20 = -torch.sin(beta)
        R21 = torch.sin(alpha)*torch.cos(beta)
        R22 = torch.cos(alpha)*torch.cos(beta)
        R23 = x[:,5:6]
        combined = torch.cat([R00, R01, R02, R03, R10, R11, R12, R13, R20, R21, R22, R23], dim=1)
        result = combined.view(-1, 3, 4)
        # 3) Prepare the fixed 4th row [0, 0, 0, 1] as shape [N, 1, 4]
        #    We'll broadcast (expand) this row for each of the N samples.
        fixed_row = torch.tensor([0, 0, 0, 1]).view(1, 1, 4)
        fixed_row = fixed_row.expand(result.shape[0], 1, 4)  # shape: [N, 1, 4]

        # 4) Concatenate the top 3 rows and the fixed 4th row => [N, 4, 4]
        result = torch.cat([result, fixed_row], dim=1)  # shape: [N, 4, 4]
        return result 

model = MyModel()
my_input = torch.Tensor(np.array([[0,0,0,0,0,0]])) # 1x6
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.03)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(my_input,), method="backward")
print(lb, ub )