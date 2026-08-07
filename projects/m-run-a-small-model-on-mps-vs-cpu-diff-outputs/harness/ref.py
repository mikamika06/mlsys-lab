import torch
import time
import re

def compare_outputs(model, x):
    cpu_model = model.to('cpu')
    with torch.no_grad():
        out_cpu = cpu_model(x.to('cpu'))
        if torch.backends.mps.is_available():
            mps_model = model.to('mps')
            out_mps = mps_model(x.to('mps'))
            diff = torch.abs(out_cpu - out_mps.to('cpu'))
        else:
            diff = torch.abs(out_cpu - out_cpu)
    return float(torch.max(diff).item())

def measure_staging_cost(sizes):
    costs = {}
    for size in sizes:
        x = torch.randn(size, size)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            t0 = time.time()
            _ = x.to('mps')
            torch.mps.synchronize()
            t1 = time.time()
            costs[size] = t1 - t0
        else:
            t0 = time.time()
            _ = x.to('cpu')
            t1 = time.time()
            costs[size] = t1 - t0
    return costs

def extract_unsupported_op(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return None
    except Exception as e:
        msg = str(e)
        match = re.search(r"The operator '([^']+)' is not implemented for the MPS device", msg)
        if match:
            return match.group(1)
        if "NotImplementedError" in type(e).__name__ or "MPS" in msg:
            parts = msg.split("'")
            if len(parts) >= 2:
                return parts[1]
        raise e
