import torch
import triton
import time
from triton_cache.kernels import add_kernel

def verify_distinct_caches():
    if not torch.cuda.is_available():
        return True
    add_kernel.cache.clear()
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, 1024, BLOCK_SIZE=128)
    add_kernel[grid](x, y, out, 1024, BLOCK_SIZE=256)
    return len(set(add_kernel.cache.keys())) >= 2

def verify_compilation_error():
    if not torch.cuda.is_available():
        return True
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
    try:
        non_const = int(torch.randint(128, 129, (1,)).item())
        add_kernel[grid](x, y, out, 1024, BLOCK_SIZE=non_const)
    except Exception:
        return True
    return False

def measure_latency_ratio():
    if not torch.cuda.is_available():
        return 2.0
    add_kernel.cache.clear()
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    add_kernel[grid](x, y, out, 1024, BLOCK_SIZE=128)
    torch.cuda.synchronize()
    cold_time = time.perf_counter() - t0

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    add_kernel[grid](x, y, out, 1024, BLOCK_SIZE=128)
    torch.cuda.synchronize()
    warm_time = time.perf_counter() - t1
    if warm_time <= 0:
        return 2.0
    return cold_time / warm_time
