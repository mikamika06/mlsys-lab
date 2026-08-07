import time
import torch
from torch.utils.checkpoint import checkpoint


def measure_checkpoint_overhead(module, x):
    model_normal = module
    model_ckpt = module

    x_normal = x.detach().clone().requires_grad_(True)

    start = time.perf_counter()
    out = model_normal(x_normal)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize() if x.is_cuda else None
    normal_time = time.perf_counter() - start

    x_ckpt = x.detach().clone().requires_grad_(True)

    def run_ckpt(inputs):
        return checkpoint(model_ckpt, inputs, use_reentrant=False)

    start = time.perf_counter()
    out_ckpt = run_ckpt(x_ckpt)
    loss_ckpt = out_ckpt.sum()
    loss_ckpt.backward()
    torch.cuda.synchronize() if x.is_cuda else None
    ckpt_time = time.perf_counter() - start

    ratio = ckpt_time / max(normal_time, 1e-6)
    return {"normal_time": normal_time, "ckpt_time": ckpt_time, "ratio": ratio}
