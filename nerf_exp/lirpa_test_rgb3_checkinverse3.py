import torch 
import numpy as np 
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

class InverseModel(torch.nn.Module):
    def forward(self, x):
        x00 = x[:,0:1,0]
        x01 = x[:,0:1,1]
        x10 = x[:,1:2,0]
        x11 = x[:,1:2,1]

        det = x00*x11-x01*x10

        xinv00 = x11/det 
        xinv01 = -x01/det 
        xinv10 = -x10/det 
        xinv11 = x00/det 

        res = torch.cat([xinv00, xinv01, xinv10, xinv11], dim=1).reshape((-1,2,2))
        return res 

class EpsModel(torch.nn.Module):
    def __init__(self, A0inv):
        super().__init__()
        self.A0inv = A0inv
        self.device = torch.device('cuda')

    def to(self, *args, **kwargs):
        # Move parameters and buffers
        super(EpsModel, self).to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device', None)
        if device is not None:
            # Move more custom tensors
            self.A0inv = self.A0inv.to(device)
            # Update device attribute
            self.device = torch.device(device)
        return self  # Important: Return self to allow method chaining
    
    def forward(self, x:torch.Tensor):
        # ft = self.A0inv\
        #     -self.A0inv@x@self.A0inv\
        #     +self.A0inv@x@self.A0inv@x@self.A0inv\
        #     -self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv\
        #     +self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv\
        #     -self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv\
        #     +self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv\
        #     -self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv
        # ft = self.A0inv@\
        #     (torch.eye(2).to(x.device)-x@self.A0inv@\
        #     (torch.eye(2).to(x.device)+x@self.A0inv@\
        #     (torch.eye(2).to(x.device)-x@self.A0inv@\
        #     (torch.eye(2).to(x.device)+x@self.A0inv@\
        #     (torch.eye(2).to(x.device)-x@self.A0inv@\
        #     (torch.eye(2).to(x.device)+x@self.A0inv@\
        #     (torch.eye(2).to(x.device)-x@self.A0inv@\
        #     (torch.eye(2).to(x.device)+x@self.A0inv@\
        #     (torch.eye(2).to(x.device)-x@self.A0inv)))))))))
        # ft = self.A0inv@\
        #     (torch.eye(2).to(x.device)-(x@self.A0inv))@\
        #     (torch.eye(2).to(x.device)+(x@self.A0inv@x@self.A0inv))@\
        #     (torch.eye(2).to(x.device)+(x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv))@\
        #     (torch.eye(2).to(x.device)+(x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv))@\
        #     (torch.eye(2).to(x.device)+(x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv@x@self.A0inv))
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
            (torch.eye(2).to(x.device)+t16)
            # (torch.eye(2).to(x.device)+t32)@\
            # (torch.eye(2).to(x.device)+t64)
        
        return ft

if __name__ == "__main__":

    # A_lb = torch.Tensor([[[
    #     [58951.977, -2156.779],
    #     [-2156.7783,  86873.53],
    # ]]])
    # A_ub = torch.Tensor([[[
    #     [142558.67, 718.92676],
    #     [718.92676, 142558.69]
    # ]]])

    A_lb = torch.Tensor([[[
        [58951.977, -2156.779],
        [-2156.7783,  86873.53],
    ]]])
    A_ub = torch.Tensor([[[
        [142558.67, 718.92676],
        [718.92676, 142558.69]
    ]]])
    # A_lb = torch.Tensor([[[
    #     [0.9,-0.1],
    #     [-0.1,0.9]
    # ]]])
    # A_ub = torch.Tensor([[[
    #     [1.1,0.1],
    #     [0.1,1.1]
    # ]]])

    A0 = ((A_lb+A_ub)/2).squeeze(0)
    A0inv = torch.inverse(A0)
    delta = (A_ub-A_lb)/2
    eps = torch.norm(delta@A0inv)
    print(eps)
    delta0 = torch.zeros(delta.shape)

    modeleps = EpsModel(A0inv)
    ptb_A = PerturbationLpNorm(
        norm=np.inf,
        x_L = -delta, 
        x_U = delta,  
    )
    my_input = BoundedTensor(delta0, ptb_A)
    model_bounded = BoundedModule(modeleps, delta0, device = 'cpu', bound_opts={'conv_mode': 'matrix'})
    prediction = model_bounded(my_input)
    model_bounded.visualize('inverse_new')
    
    lb_eps, ub_eps = model_bounded.compute_bounds(x=(my_input, ), method='alpha-crown')
    # print(lb_eps, ub_eps)
    # if eps>1:
    #     print("EPS constraint Violated")

    T = torch.norm(A0inv)*eps**16/(1-eps)
    print(T)

    res_lb = lb_eps-torch.Tensor([[
        [T, T/np.sqrt(2)],
        [T/np.sqrt(2), T]
    ]])
    res_ub = ub_eps+torch.Tensor([[
        [T, T/np.sqrt(2)],
        [T/np.sqrt(2), T]
    ]])
    print("ours: ", res_lb, res_ub)

    res_lb = res_lb.detach().cpu().numpy()
    res_ub = res_ub.detach().cpu().numpy()

    modelinv = InverseModel()
    ptb_A = PerturbationLpNorm(
        norm=np.inf,
        x_L = A_lb, 
        x_U = A_ub,  
    )
    inp_A = BoundedTensor(A0, ptb_A)
    model_bounded = BoundedModule(modelinv, delta0, device = 'cpu', bound_opts={'conv_mode': 'matrix'})
    lb_inv, ub_inv = model_bounded.compute_bounds(x=(inp_A, ), method='alpha-crown')
    print("crown: ", lb_inv, ub_inv)


    A_lb = A_lb.detach().cpu().numpy()
    A_ub = A_ub.detach().cpu().numpy()
    emp_lb = np.ones(res_lb.shape)*np.inf
    emp_ub = -np.ones(res_ub.shape)*np.inf
    for i in range(1000):
        A = np.random.uniform(A_lb, A_ub)
        Ainv = np.linalg.inv(A)
        emp_lb = np.minimum(emp_lb, Ainv)
        emp_ub = np.maximum(emp_ub, Ainv)
        # if np.any(Ainv<res_lb) or np.any(Ainv>res_ub):
        #     print("bound violated")
    print("emp: ", emp_lb, emp_ub)

