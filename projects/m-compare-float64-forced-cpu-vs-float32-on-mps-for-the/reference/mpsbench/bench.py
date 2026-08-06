import time
import torch


def time_execution(fn, device_str):
    if device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    t0 = time.perf_counter()
    res = fn()
    if device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    t1 = time.perf_counter()
    return res, t1 - t0
