import torch
from cpuharness.harness import measure_latencies

def compare_cpu_modes(model, inputs):
    eager_lat = measure_latencies(model, inputs)

    modes = ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]
    results = {"eager": eager_lat}

    for mode in modes:
        try:
            compiled = torch.compile(model, mode=mode)
            lat = measure_latencies(compiled, inputs)
            results[mode] = lat
        except Exception:
            results[mode] = float("inf")

    return results
