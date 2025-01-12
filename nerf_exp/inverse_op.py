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

class InverseModelLb(torch.nn.Module):
    def __init__(
        self,
        A0,  # 1*2*2
        device = torch.device('cuda')
    ):
        super().__init__()  # Initialize the base class first
        self.device = device
        self.A0 = A0.to(self.device)          
        self.A0_inv = torch.inverse(self.A0)

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(InverseModelLb, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move more custom tensors
            self.A0 = self.A0.to(device)
            self.A0_inv = self.A0_inv.to(device)
            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining

    def forward(self, x):
        res = self.A0_inv \
            - self.A0_inv@(x-self.A0)@self.A0_inv
        return res
            # + self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv\
            # - self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv

class InverseModelUb(torch.nn.Module):
    def __init__(
        self,
        A0,  # 1*2*2
        device = torch.device('cuda')
    ):
        super().__init__()  # Initialize the base class first
        self.device = device
        self.A0 = A0.to(self.device)          
        self.A0_inv = torch.inverse(self.A0)

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(InverseModelUb, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move more custom tensors
            self.A0 = self.A0.to(device)
            self.A0_inv = self.A0_inv.to(device)
            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining

    def forward(self, x):
        res = self.A0_inv \
            - self.A0_inv@(x-self.A0)@self.A0_inv\
            + (((
                self.A0_inv.transpose(-1,-2)@(x-self.A0).transpose(-1,-2)).transpose(-1,-2)@(x-self.A0)
                ).transpose(-1,-2)@self.A0_inv.transpose(-1,-2)
            ).transpose(-1,-2)@self.A0_inv
            # - self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv\
            # + self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv@(x-self.A0)@self.A0_inv
        return res 

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

    def forward(self, x):
        res = torch.inverse(x)
        return res

    def bound_backward(self, last_lA, last_uA, x, *args, **kwargs):
        A_lb, A_ub = x.lower, x.upper
        A0 = (A_lb+A_ub)/2

        model_inverse_lb = InverseModelLb(A0)
        model_inverse_ub = InverseModelUb(A0)
    
        ptb_A = PerturbationLpNorm(
            norm=np.inf, 
            x_L=A_lb.to(model_inverse_lb.device),
            x_U=A_ub.to(model_inverse_lb.device),
        )
        my_input = BoundedTensor(A0, ptb_A)
        model_inverselb_bounded = BoundedModule(model_inverse_lb, A0, device=model_inverse_lb.device, bound_opts={'conv_mode': 'matrix'})
        required_A = defaultdict(set)
        required_A[model_inverselb_bounded.output_name[0]].add(model_inverselb_bounded.input_name[0])
        lb_inverselb, ub_inverselb, A_inverselb = model_inverselb_bounded.compute_bounds(x=(my_input, ), method='crown', return_A=True, needed_A_dict=required_A, can_skip=True)
        
        # b_inverselb, ub_inverselb = model_inverselb_bounded.compute_bounds(x=(my_input, ), method='crown')
        model_inverseub_bounded = BoundedModule(model_inverse_ub, A0, device=model_inverse_ub.device, bound_opts={'conv_mode': 'matrix'})
        required_A = defaultdict(set)
        required_A[model_inverseub_bounded.output_name[0]].add(model_inverseub_bounded.input_name[0])
        lb_inverseub, ub_inverseub, A_inverseub = model_inverseub_bounded.compute_bounds(x=(my_input, ), method='crown', return_A=True, needed_A_dict=required_A, can_skip=True)       

        lAlb: torch.Tensor = A_inverselb[model_inverselb_bounded.output_name[0]][model_inverselb_bounded.input_name[0]]['lA']
        lbiaslb: torch.Tensor = A_inverselb[model_inverselb_bounded.output_name[0]][model_inverselb_bounded.input_name[0]]['lbias']
        uAub: torch.Tensor = A_inverseub[model_inverseub_bounded.output_name[0]][model_inverseub_bounded.input_name[0]]['uA']
        ubiasub: torch.Tensor = A_inverseub[model_inverseub_bounded.output_name[0]][model_inverseub_bounded.input_name[0]]['ubias']
        # print(lb_inverselb, ub_inverseub)

        lA = last_lA*lAlb.transpose(0,1)
        lbias = last_lA.sum(dim=list(range(2, last_lA.ndim)))*lbiaslb.transpose(0,1)

        uA = last_uA*uAub.transpose(0,1)
        ubias = last_uA.sum(dim=list(range(2, last_uA.ndim)))*ubiasub.transpose(0,1)

        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        A_lb, A_ub = v[0]
        A0 = (A_lb+A_ub)/2

        model_inverse_lb = InverseModelLb(A0)
        model_inverse_ub = InverseModelUb(A0)
    
        ptb_A = PerturbationLpNorm(
            norm=np.inf, 
            x_L=A_lb.to(model_inverse_lb.device),
            x_U=A_ub.to(model_inverse_lb.device),
        )
        my_input = BoundedTensor(A0, ptb_A)
        model_inverselb_bounded = BoundedModule(model_inverse_lb, A0, device=model_inverse_lb.device, bound_opts={'conv_mode': 'matrix'})
        # model_inverselb_bounded.visualize('inverse_lb')
        lb_inverselb, ub_inverselb = model_inverselb_bounded.compute_bounds(x=(my_input, ), method='crown', can_skip=True)
        model_inverseub_bounded = BoundedModule(model_inverse_ub, A0, device=model_inverse_ub.device, bound_opts={'conv_mode': 'matrix'})
        lb_inverseub, ub_inverseub = model_inverseub_bounded.compute_bounds(x=(my_input, ), method='crown', can_skip=True)       
        # print(lb_inverselb, ub_inverseub)
        return Interval.make_interval(lb_inverselb, ub_inverseub) 
   
""" Step 4: Register the custom operator """
register_custom_op("custom::Inverse", BoundInverse)

if __name__ == "__main__":
    class InverseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.inv_op = Inverse()
            self.const = torch.Tensor([[
                [2,0],
                [0,3]
            ]]).to('cuda')
        
        def to(self, *args, **kwargs):
            # Move parameters and buffers
            super(InverseModel, self).to(*args, **kwargs)
            device = args[0] if args else kwargs.get('device', None)
            if device is not None:
                # Move more custom tensors
                self.const = self.const.to(device)
            return self  # Important: Return self to allow method chaining

        def forward(self, x):
            x = x+self.const
            # res = x
            res = torch.pow(x,6)
            res = self.inv_op(x)
            res = torch.pow(res, 4)
            res = torch.abs(res)
            return res    

    A_lb = torch.Tensor([[[
        [58951.977, -2156.779],
        [-2156.7783,  86873.53],
    ],[
        [96794.266, -718.9259],
        [-718.9259,  96794.3],
    ],[
        [107970.2, -2156.7788],
        [-2156.779, 71892.664],
    ]]]).to('cuda')
    A_ub = torch.Tensor([[[
        [93460.44, 2156.7783],
        [2156.7793, 130492.9]
    ],[
        [142558.67, 718.92676],
        [718.92676, 142558.69]
    ],[
        [160912.1, 2156.7793],
        [2156.7793, 104244.3]
    ]]]).to('cuda')
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
    # lb_inverse, ub_inverse = model_inverse_bounded.compute_bounds(x=(my_input, ), method='ibp')
    lb_inverse, ub_inverse = model_inverse_bounded.compute_bounds(x=(my_input, ), method='crown')
    print(lb_inverse, ub_inverse)

