import torch

P = 100
N = 500

# Non-symmetric step_res (1x3x3)
step_res = torch.rand((1,N,N))  # shape (1, 3, 3)
alpha_bound = torch.rand((1, P, N, 1))  # shape (1, 2, 3, 1)

# Original code output
def computeT_original(alpha_bound, step_res):
    T = torch.zeros(alpha_bound.shape).to(alpha_bound.device)
    for i in range(step_res.shape[2]):
        T[0,:,i,:] = (torch.ones((1,1,1,1)).to(alpha_bound.device)-(alpha_bound*step_res[:,None,i,:,None].to(alpha_bound.device))).prod(dim=2)
    return T

# Optimized code output
def computeT_optimized(alpha_bound, step_res):
    A = alpha_bound.squeeze(0).squeeze(-1)  # (P, N)
    S = step_res.squeeze(0)                 # (N, N)
    P, N = A.shape[0], S.shape[0]
    device = A.device
    result = torch.ones((P, N), device=device)
    for j in range(N):
        A_j = A[:, j]          # (P,)
        S_j = S[:, j]          # (N,)
        term = 1 - torch.outer(A_j, S_j)  # (P, N)
        result *= term
    return result.unsqueeze(0).unsqueeze(-1)

T_original = computeT_original(alpha_bound, step_res)
T_optimized = computeT_optimized(alpha_bound, step_res)

# Check equivalence
print(torch.allclose(T_original, T_optimized))  # Output: True