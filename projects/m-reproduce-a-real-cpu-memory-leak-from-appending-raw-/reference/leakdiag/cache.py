import torch


def simulate_kv_cache(steps, reset):
    cache = []
    peak_memory = 0
    for i in range(steps):
        tokens = torch.randn(1, 16, 64)
        if reset:
            cache = [tokens]
        else:
            cache.append(tokens)
        current_mem = sum(t.nelement() * t.element_size() for t in cache)
        if current_mem > peak_memory:
            peak_memory = current_mem
    return float(peak_memory)
