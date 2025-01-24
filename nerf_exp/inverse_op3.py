""" A example for custom operators.

In this example, we create a custom operator called "PlusConstant", which can
be written as "f(x) = x + c" for some constant "c" (an attribute of the operator).
"""
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
from auto_LiRPA.operators import Bound
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
import numpy as np 
from auto_LiRPA.operators.base import Interval
from typing import List 
from collections import defaultdict

class EpsModel(torch.nn.Module):
    def __init__(self, A0inv, device):
        super().__init__()
        A0inv = A0inv.detach()
        self.register_buffer("A0inv", A0inv)
        self.device = device

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(EpsModel, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move more custom tensors
            # self.A0inv = self.A0inv.to(device)
            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining
    
    def forward(self, x:torch.Tensor):
        t2 = x@self.A0inv
        t4 = t2@t2 
        t8 = t4@t4
        t16 = t8@t8
        t32 = t16@t16
        t64 = t32@t32
        ft = self.A0inv@\
            (torch.eye(2).to(x.device)-t2)@\
            (torch.eye(2).to(x.device)+t4)@\
            (torch.eye(2).to(x.device)+t8)@\
            (torch.eye(2).to(x.device)+t16)#@\
            # (torch.eye(2).to(x.device)+t32)#@\
            # (torch.eye(2).to(x.device)+t64)
        
        return ft

""" Step 1: Define a `torch.autograd.Function` class to declare and implement the
computation of the operator. """
class InverseOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, A):
        """ In this function, define the arguments and attributes of the operator.
        "custom::PlusConstant" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::Inverse', A)

    @staticmethod
    def forward(ctx, x):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = x + c in this case. """
        res = torch.inverse(x)
        return res

""" Step 2: Define a `torch.nn.Module` class to declare a module using the defined
custom operator. """
class Inverse(nn.Module):
    def __init__(self, const=1):
        super().__init__()

    def forward(self, x):
        """ Use `PlusConstantOp.apply` to call the defined custom operator. """
        return InverseOp.apply(x)

""" Step 3: Implement a Bound class to support bound computation for the new operator. """
class BoundInverse(Bound):
    def __init__(self, attr, inputs, output_index, options):
        """ `const` is an attribute and can be obtained from the dict `attr` """
        super().__init__(attr, inputs, output_index, options)
        self.requires_input_bounds = [0]
        self.counter = 0
        self.stored_bounds = None 
        self.stored_inp_lb = None 
        self.stored_inp_ub = None 
        self.alpha_counter = 0

    def forward(self, x):
        res = torch.inverse(x)
        return res

    def bound_backward(self, last_lA, last_uA, x, *args, **kwargs):
        A_lb, A_ub = x.lower.clone().detach(), x.upper.clone().detach()
        A0 = (A_lb+A_ub)/2
        delta = (A_ub-A_lb)/2
        A0inv = torch.inverse(A0).squeeze()
        val = torch.norm(delta@A0inv)   
        T = torch.norm(A0inv)*val**16/(1-val)
        T_mat = torch.Tensor([[
            [T, T/np.sqrt(2)],
            [T/np.sqrt(2), T]
        ]]).to(A0.device)

        # model_inverse_lb = InverseModelLb(A0)
        # model_inverse_ub = InverseModelUb(A0)
        eps_model = EpsModel(A0inv, device = A0inv.device)
    
        ptb_delta = PerturbationLpNorm(
            norm=np.inf, 
            x_L=-delta.to(eps_model.device),
            x_U=delta.to(eps_model.device),
        )
        my_input = BoundedTensor(A0, ptb_delta)
        model_eps_bounded = BoundedModule(
            eps_model, A0, 
            device=eps_model.device, 
            bound_opts={
                'conv_mode': 'matrix', 
                'optimize_bound_args': {'iteration': 10}
            },
        )
        required_A = defaultdict(set)
        required_A[model_eps_bounded.output_name[0]].add(model_eps_bounded.input_name[0])
        # with torch.no_grad():
        # if True:
        if self.counter%30==0 or self.stored_bounds is None or torch.any(A_lb<self.stored_inp_lb) or torch.any(A_ub>self.stored_inp_ub):
            print(f"####### Bound Backward Inverse Alpha: {self.counter}, {self.alpha_counter}")
            lb_eps, ub_eps, A_eps = model_eps_bounded.compute_bounds(
                x=(my_input, ), method='alpha-crown', return_A=True, needed_A_dict=required_A,
            )
            self.stored_bounds = A_eps 
            self.stored_inp_lb = A_lb 
            self.stored_inp_ub = A_ub
            self.alpha_counter+= 1
        else:
            print(f"####### Bound Backward Inverse: {self.counter}, {self.alpha_counter}")
            A_eps = self.stored_bounds
        self.counter += 1 
        EpsAlb: torch.Tensor = A_eps[model_eps_bounded.output_name[0]][model_eps_bounded.input_name[0]]['lA']
        Epsbiaslb: torch.Tensor = A_eps[model_eps_bounded.output_name[0]][model_eps_bounded.input_name[0]]['lbias']
        EpsAub: torch.Tensor = A_eps[model_eps_bounded.output_name[0]][model_eps_bounded.input_name[0]]['uA']
        Epsbiasub: torch.Tensor = A_eps[model_eps_bounded.output_name[0]][model_eps_bounded.input_name[0]]['ubias']
        EpsAlb_const = EpsAlb.clone().detach()
        Epsbiaslb_const = Epsbiaslb.clone().detach()
        EpsAub_const = EpsAub.clone().detach()
        Epsbiasub_const = Epsbiasub.clone().detach()

        A0 = A0.squeeze()
        A0_const = A0.clone().detach()
        T_mat_const = T_mat.clone().detach()

        if last_lA is None:
            lA = None 
            lbias = 0 
        else:
            last_lA_view = last_lA.view((last_lA.shape[0], last_lA.shape[1], -1))
            lA = torch.einsum('ijk,jkab->ijab', last_lA_view, EpsAlb_const)
            lbias = -torch.einsum("xyuv,yuv->xy", lA, A0_const)
            lbias = lbias+torch.einsum('ixa,xa->ix', last_lA_view, Epsbiaslb_const)
            lbias = lbias-torch.einsum('ixab,xab->ix', last_lA, T_mat_const)

        if last_uA is None:
            uA = None 
            ubias = 0
        else:
            last_uA_view = last_uA.view((last_uA.shape[0], last_uA.shape[1], -1))
            uA = torch.einsum('ijk,jkab->ijab', last_uA_view, EpsAub_const)
            ubias = -torch.einsum("xyuv,yuv->xy", uA, A0_const)
            ubias = ubias+torch.einsum('ixa,xa->ix', last_uA_view, Epsbiasub_const)
            ubias = ubias-torch.einsum('ixab,xab->ix', last_uA, T_mat_const)

        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        A_lb, A_ub = v[0]
        A0 = (A_lb+A_ub)/2
        delta = (A_ub-A_lb)/2
        A0inv = torch.inverse(A0)
        val = torch.norm(delta@A0inv)   
        T = torch.norm(A0inv)*val**16/(1-val)
        T_mat = torch.Tensor([[
            [T, T/np.sqrt(2)],
            [T/np.sqrt(2), T]
        ]]).to(A0.device)

        eps_model = EpsModel(A0inv, A0inv.device)
    
        ptb_delta = PerturbationLpNorm(
            norm=np.inf, 
            x_L=-delta.to(eps_model.device),
            x_U=delta.to(eps_model.device),
        )
        my_input = BoundedTensor(torch.zeros(delta.shape).to(delta.device), ptb_delta)
        model_eps_bounded = BoundedModule(eps_model, A0, device=eps_model.device, bound_opts={'conv_mode': 'matrix'})
        required_A = defaultdict(set)
        required_A[model_eps_bounded.output_name[0]].add(model_eps_bounded.input_name[0])
        lb_eps, ub_eps = model_eps_bounded.compute_bounds(x=(my_input, ), method='ibp') 

        lb_eps = lb_eps - T_mat 
        ub_eps = ub_eps + T_mat
    
        return Interval.make_interval(lb_eps, ub_eps) 
   
""" Step 4: Register the custom operator """
register_custom_op("custom::Inverse", BoundInverse)

if __name__ == "__main__":
    class InverseModel(nn.Module):
        def __init__(self, device = 'cpu'):
            super().__init__()
            self.inv_op = Inverse()
            self.const = torch.Tensor([[[
                [2,0],
                [0,3]
            ]]]).to(device)
        
        def to(self, *args, **kwargs):
            # Move parameters and buffers
            super(InverseModel, self).to(*args, **kwargs)
            device = args[0] if args else kwargs.get('device', None)
            if device is not None:
                # Move more custom tensors
                self.const = self.const.to(device)
            return self  # Important: Return self to allow method chaining

        def forward(self, x):
            # res = x
            # res = x*self.const
            # res = torch.pow(x,3)
            res = self.inv_op(x)
            # res = res-1
            # res = res*self.const
            # res = torch.abs(res)
            return res    

    # A_lb = torch.Tensor([[[
    #     [58951.977, -2156.779],
    #     [-2156.779,  86873.53],
    # ],[
    #     [96794.266, -718.9259],
    #     [-718.9259,  96794.3],
    # ],[
    #     [107970.2, -2156.7788],
    #     [-2156.7788, 71892.664],
    # ]]]).to('cuda')
    # A_ub = torch.Tensor([[[
    #     [93460.44, 2156.7783],
    #     [2156.7783, 130492.9]
    # ],[
    #     [142558.67, 718.92676],
    #     [718.92676, 142558.69]
    # ],[
    #     [160912.1, 2156.7793],
    #     [2156.7793, 104244.3],
    # ]]]).to('cuda')
    A_lb = torch.Tensor([[
        [0.9, -0.1],
        [-0.1, 0.9],
    ]]).to('cuda')
    A_ub = torch.Tensor([[
        [1.1, 0.1],
        [0.1, 1.1],
    ]]).to('cuda')
    
    # A_lb = torch.Tensor([[[
    #     [58951.977, -2156.779],
    #     [-2156.7783,  86873.53],
    # ],[
    #     [96794.266, -718.9259],
    #     [-718.9259,  96794.3],
    # ],[
    #     [107970.2, -2156.7788],
    #     [-2156.779, 71892.664],
    # ]]]).to('cuda')
    # A_ub = torch.Tensor([[[
    #     [93460.44, 2156.7783],
    #     [2156.7793, 130492.9]
    # ],[
    #     [142558.67, 718.92676],
    #     [718.92676, 142558.69]
    # ],[
    #     [160912.1, 2156.7793],
    #     [2156.7793, 104244.3]
    # ]]]).to('cuda')
    # A_lb = torch.Tensor([[
    #     [0.9,0],
    #     [0,0.9]
    # ]]).to('cuda')
    # A_ub = torch.Tensor([[
    #     [1.1,0],
    #     [0,1.1]
    # ]]).to('cuda')
    A0 = ((A_lb+A_ub)/2)

    model = InverseModel()

    ptb_A = PerturbationLpNorm(
        norm=np.inf, 
        x_L=A_lb,
        x_U=A_ub,
    )
    my_input = BoundedTensor(A0, ptb_A)
    model_inverse_bounded = BoundedModule(model, A0, device=torch.device('cuda'), bound_opts={'conv_mode': 'matrix'})
    prediction = model_inverse_bounded(my_input)
    model_inverse_bounded.visualize('inverse')
    
    lb_inverse, ub_inverse = model_inverse_bounded.compute_bounds(x=(my_input, ), method='alpha-crown')
    print(lb_inverse, ub_inverse)
    