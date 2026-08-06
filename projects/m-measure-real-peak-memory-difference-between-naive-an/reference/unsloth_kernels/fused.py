import torch

def fused_chunked_reduction_grad(x_data):
    x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
    w = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    out_naive = (x * w).sum()
    out_naive.backward()
    grad_naive = x.grad.clone()
    x.grad = None
    w.grad = None
    chunks = torch.split(x, max(1, x.shape[0] // 2), dim=0)
    out_fused = sum((c * w).sum() for c in chunks)
    out_fused.backward()
    grad_fused = x.grad.clone()
    return grad_fused.detach().numpy(), grad_naive.detach().numpy()
