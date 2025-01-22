import torch 
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

class MyModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A0 = torch.rand((100,1,2))

    def forward(self, x: torch.Tensor):
        res = torch.matmul(self.A0, x)
        return res 

x = torch.Tensor([[
    [1,0],
    [0,1]
]]).repeat((100,1,1))
print(x.shape)
model = MyModel1()
res = model(x)
print(res.shape)
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
