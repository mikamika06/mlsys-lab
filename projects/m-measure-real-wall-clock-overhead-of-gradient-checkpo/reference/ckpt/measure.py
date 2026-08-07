import time
import torch


def measure_overhead(module_no_ckpt, module_ckpt, x, num_warmup=3, num_iters=10):
    for _ in range(num_warmup):
        out = module_no_ckpt(x)
        out.sum().backward()
        module_no_ckpt.zero_grad()
        out_c = module_ckpt(x)
        out_c.sum().backward()
        module_ckpt.zero_grad()
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = module_no_ckpt(x)
        out.sum().backward()
        module_no_ckpt.zero_grad()
    torch.cuda.synchronize() if x.is_cuda else None
    t_no_ckpt = time.perf_counter() - t0
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out_c = module_ckpt(x)
        out_c.sum().backward()
        module_ckpt.zero_grad()
    torch.cuda.synchronize() if x.is_cuda else None
    t_ckpt = time.perf_counter() - t0
    return {"t_no_ckpt": t_no_ckpt, "t_ckpt": t_ckpt, "ratio": t_ckpt / max(t_no_ckpt, 1e-9)}
