import numpy as np


def simulate_hit_rate(requests, pool_size):
    cache = []
    hits = 0
    for r in requests:
        if r in cache:
            hits += 1
            cache.remove(r)
            cache.append(r)
        else:
            if len(cache) >= pool_size:
                cache.pop(0)
            cache.append(r)
    return float(hits) / float(len(requests)) if requests else 0.0


def validate_adapters(adapters, limits):
    max_rank = limits.get("max_rank", 64)
    max_memory = limits.get("max_memory_mb", 1024)
    total_mem = sum(a.get("memory_mb", 0) for a in adapters)
    if total_mem > max_memory:
        raise ValueError("exceeds memory limit")
    for a in adapters:
        if a.get("rank", 0) > max_rank:
            raise ValueError("exceeds rank limit")
    return {
        "max_loras": len(adapters),
        "max_lora_rank": max((a.get("rank", 0) for a in adapters), default=0),
        "lora_slots": len(adapters)
    }


def lora_forward(base_w, lora_a, lora_b, alpha, x):
    scaling = alpha / lora_a.shape[0] if lora_a.shape[0] > 0 else 1.0
    delta = (x @ lora_a) @ lora_b * scaling
    return (x @ base_w) + delta
