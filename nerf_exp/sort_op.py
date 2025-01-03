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

""" Step 1: Define a `torch.autograd.Function` class to declare and implement the
computation of the operator. """
class SortArrayOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, order_array, sort_array):
        """ In this function, define the arguments and attributes of the operator.
        "custom::PlusConstant" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::SortArray', order_array, sort_array)

    @staticmethod
    def forward(ctx, x, y):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = x + c in this case. """
        order = torch.argsort(x, dim=2).squeeze()
        sorted_y = y[:,:,order,:]
        return sorted_y

""" Step 2: Define a `torch.nn.Module` class to declare a module using the defined
custom operator. """
class SortArray(nn.Module):
    def __init__(self, const=1):
        super().__init__()

    def forward(self, x, y):
        """ Use `PlusConstantOp.apply` to call the defined custom operator. """
        return SortArrayOp.apply(x, y)

""" Step 3: Implement a Bound class to support bound computation for the new operator. """
class BoundSortArray(Bound):
    def __init__(self, attr, inputs, output_index, options):
        """ `const` is an attribute and can be obtained from the dict `attr` """
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x, y):
        order = torch.argsort(x, dim=2)
        sorted_y = y[:,:,order,:]
        return sorted_y

    def bound_backward(self, last_lA, last_uA, x, *args, **kwargs):
        # """ Backward mode bound propagation """
        # print('Calling bound_backward for custom::PlusConstant')
        # def _bound_oneside(last_A):
        #     # If last_lA or last_uA is None, it means lower or upper bound
        #     # is not required, so we simply return None.
        #     if last_A is None:
        #         return None, 0
        #     # The function f(x) = x + c is a linear function with coefficient 1.
        #     # Then A · f(x) = A · (x + c) = A · x + A · c.
        #     # Thus the new A matrix is the same as the last A matrix:
        #     A = last_A
        #     # For bias, compute A · c and reduce the dimensions by sum:
        #     bias = last_A.sum(dim=list(range(2, last_A.ndim))) * self.const
        #     return A, bias
        # lA, lbias = _bound_oneside(last_lA)
        # uA, ubias = _bound_oneside(last_lA)
        # return [(lA, uA)], lbias, ubias
        raise NotImplementedError

    def interval_propagate(self, *v):
        # """ IBP computation """
        # print('Calling interval_propagate for custom::PlusConstant')
        # # Interval bound of the input
        # h_L, h_U = v[0]
        # # Since this function is monotonic, we can get the lower bound and upper bound
        # # by applying the function on h_L and h_U respectively.
        # lower = h_L + self.const
        # upper = h_U + self.const
        # return lower, upper
        lb_depth, ub_depth = v[0]
        y_L, y_U = v[1]

        bounds_y = torch.cat((y_L, y_U), dim=0) 

        lb_depth = lb_depth.detach().cpu().numpy().reshape((1,-1))
        ub_depth = ub_depth.detach().cpu().numpy().reshape((1,-1))    
        bounds_depth = np.vstack((lb_depth, ub_depth)).T
        bounds_depth = bounds_depth.tolist()
        bounds_depth = [elem+[i] for i, elem in enumerate(bounds_depth)]
        sorted_bounds = self._sort_bounds(bounds_depth)
        set_order = self._get_set_order(sorted_bounds)
        set_sorted_val = self._apply_set_order(set_order, bounds_y)

        sorted_L = set_sorted_val[0:1]
        sorted_U = set_sorted_val[1:2]
    
        return Interval.make_interval(sorted_L, sorted_U)

    def _sort_bounds(self, bounds:List[List[float]]):
        if len(bounds) == 0:
            return bounds
        elif len(bounds) == 1:
            return bounds
        else:
            pivot = int(len(bounds)/2)
            bounds_left, bounds_right = self._reorg_bounds(bounds, pivot)
            sort_left = self._sort_bounds(bounds_left)
            sort_right = self._sort_bounds(bounds_right)
            return sort_left + [bounds[pivot]] + sort_right

    def _reorg_bounds(self, bounds, pivot):
        pivot_val = bounds[pivot]
        # res = [pivot_val]
        bounds_left = []
        bounds_right = []
        for i in range(len(bounds)):
            if i!=pivot:
                val = bounds[i]
                if val[1] <= pivot_val[0]:
                    # res = [val]+res
                    bounds_left.append(val) 
                elif pivot_val[1] <= val[0]:
                    bounds_right.append(val)
                elif val[0] < pivot_val[0]:
                    bounds_left.append(val)
                else:
                    bounds_right.append(val)
        return bounds_left, bounds_right

    def _get_set_order(self, sorted_bounds):
        res_list = []
        for i in range(len(sorted_bounds)):
            bins = []
            ref_bound = sorted_bounds[i]
            for j in range(len(sorted_bounds)):
                bound = sorted_bounds[j]
                if ref_bound[0]<=bound[1]<=ref_bound[1] or \
                    ref_bound[0]<=bound[0]<=ref_bound[1] or \
                    bound[0]<=ref_bound[0]<=bound[1] or \
                    bound[0]<=ref_bound[1]<=bound[1]:
                    bins.append(bound[2])
            res_list.append(bins)
        return res_list

    def _apply_set_order(self, set_order: List, bounds: torch.Tensor):
        '''
        set_order: List with length N
        bounds: 2*256*N 
        '''
        sorted_bounds = torch.zeros(bounds.shape).to(bounds.device)
        for i in range(len(set_order)):
            for j in range(len(set_order[i])):
                if j==0:
                    sorted_bounds[:,:,i] = bounds[:,:,set_order[i][j]]
                else:
                    sorted_bounds[:,:,i] = self._bounds_union(sorted_bounds[:,:,i], bounds[:,:,set_order[i][j]])
        return sorted_bounds

    def _bounds_union(self, b1, b2):
        '''
        b1: 2*m
        b2: 2*m
        '''
        b_out = torch.zeros(b1.shape)
        b_out[0,:] = torch.min(torch.stack((b1[0,:],b2[0,:]), dim=0), dim=0).values
        b_out[1,:] = torch.max(torch.stack((b1[1,:],b2[1,:]), dim=0), dim=0).values
        return b_out 

""" Step 4: Register the custom operator """
register_custom_op("custom::SortArray", BoundSortArray)

# # Use the `PlusConstant` module in model definition
# model = nn.Sequential(
#     Flatten(),
#     nn.Linear(28 * 28, 256),
#     PlusConstant(const=1),
#     nn.Linear(256, 10),
# )
# print("Model:", model)

# test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# N = 1
# n_classes = 10
# image = test_data.data[:N].view(N,1,28,28)
# true_label = test_data.targets[:N]
# image = image.to(torch.float32) / 255.0
# if torch.cuda.is_available():
#     image = image.cuda()
#     model = model.cuda()

# lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

# eps = 0.3
# norm = float("inf")
# ptb = PerturbationLpNorm(norm = norm, eps = eps)
# image = BoundedTensor(image, ptb)
# pred = lirpa_model(image)
# label = torch.argmax(pred, dim=1).cpu().detach().numpy()

# for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
#     print("Bounding method:", method)
#     lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
#     for i in range(N):
#         print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
#         for j in range(n_classes):
#             indicator = '(ground-truth)' if j == true_label[i] else ''
#             print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
#                 j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
#     print()

