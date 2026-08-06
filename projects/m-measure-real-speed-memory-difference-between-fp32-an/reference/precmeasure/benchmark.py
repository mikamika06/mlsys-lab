import time
import torch


def measure_performance(model, inputs):
    model.eval()
    device = next(model.parameters()).device
    device_type = "cuda" if device.type == "cuda" else "cpu"

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.autocast(device_type=device_type, enabled=False):
        with torch.no_grad():
            _ = model(inputs)
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    fp32_time = t1 - t0

    fp32_mem = 0
    if device_type == "cuda":
        fp32_mem = float(torch.cuda.max_memory_allocated(device))

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t2 = time.perf_counter()
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
        with torch.no_grad():
            _ = model(inputs)
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    bf16_time = t3 - t2

    bf16_mem = 0
    if device_type == "cuda":
        bf16_mem = float(torch.cuda.max_memory_allocated(device))

    return {
        "fp32_time": float(fp32_time),
        "bf16_time": float(bf16_time),
        "fp32_memory": float(fp32_mem),
        "bf16_memory": float(bf16_mem),
        "latency_ratio": float(bf16_time / (fp32_time + 1e-9)),
    }
