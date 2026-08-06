def pack_experts(loads, num_gpus):
    indexed = sorted(enumerate(loads), key=lambda x: x[1], reverse=True)
    gpus = [[] for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus
    for idx, load in indexed:
        target = min(range(num_gpus), key=lambda g: gpu_loads[g])
        gpus[target].append(idx)
        gpu_loads[target] += load
    return gpus
