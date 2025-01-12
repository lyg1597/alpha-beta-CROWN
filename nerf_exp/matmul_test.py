import torch 
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

class MyModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A0 = torch.Tensor([
            [2,3],
            [3,2]
        ])

    def forward(self, x: torch.Tensor):
        res = self.A0@\
            (torch.eye(2).to(x.device)-(x@self.A0))@\
            (torch.eye(2).to(x.device)+(x@self.A0@x@self.A0))
        # res = self.A0\
        #     -self.A0@x@self.A0\
        #     +self.A0@x@self.A0@x@self.A0\
        #     -self.A0@x@self.A0@x@self.A0@x@self.A0\
        #     +self.A0@x@self.A0@x@self.A0@x@self.A0@x@self.A0\
        
        return res 

x = torch.Tensor([[
    [1,0],
    [0,1]
]])
model = MyModel1()
model_bounded = BoundedModule(model, (x,), device = 'cpu', bound_opts={'conv_mode': 'matrix'})
ptb = PerturbationLpNorm(
    norm=float('inf'), 
    eps = 0.1
)
inp_x = BoundedTensor(x, ptb)
res = model_bounded(inp_x)
model_bounded.visualize('matmult')

print('>>>>>> lb, ub with crown')
required_A = defaultdict(set)
required_A[model_bounded.output_name[0]].add(model_bounded.input_name[0])
lb, ub, A = model_bounded.compute_bounds(x=(inp_x, ), method = 'crown', return_A=True, needed_A_dict=required_A)
print(lb)
print(ub)
