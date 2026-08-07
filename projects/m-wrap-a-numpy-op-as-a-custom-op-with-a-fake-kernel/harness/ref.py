import torch
import numpy as np

def get_fixtures(B=2, N=4, M=5, D=3, seed=42):
    np.random.seed(seed)
    x = np.random.randn(B, N, D).astype(np.float64)
    y = np.random.randn(B, M, D).astype(np.float64)
    gamma = 0.5
    return x, y, gamma

def oracle_rbf(x, y, gamma):
    x_t = torch.from_numpy(x).requires_grad_(True)
    y_t = torch.from_numpy(y).requires_grad_(True)

    diff = x_t.unsqueeze(2) - y_t.unsqueeze(1)
    dist2 = torch.sum(diff ** 2, dim=-1)
    out = torch.exp(-gamma * dist2)

    g = torch.randn_like(out)
    out.backward(g)

    return out.detach().numpy(), x_t.grad.numpy(), y_t.grad.numpy(), g.numpy()
