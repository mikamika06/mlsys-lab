import random

def generate_placement_cases():
    random.seed(42)
    cases = []
    for _ in range(5):
        num_layers = random.randint(2, 8)
        num_experts = random.randint(4, 16)
        expert_bytes = random.randint(10, 50) * 1024 * 1024
        total_experts = num_layers * num_experts
        fit_experts = random.randint(1, total_experts)
        vram_budget = fit_experts * expert_bytes + random.randint(0, expert_bytes - 1)

        freq_log = {}
        for l in range(num_layers):
            for e in range(num_experts):
                freq_log[(l, e)] = random.randint(0, 1000)

        cases.append({
            "num_layers": num_layers,
            "num_experts": num_experts,
            "expert_bytes": expert_bytes,
            "vram_budget": vram_budget,
            "frequency_log": freq_log
        })
    return cases

def reconstruct_placement(num_layers, num_experts, expert_bytes, vram_budget, frequency_log):
    candidates = []
    for layer in range(num_layers):
        for expert in range(num_experts):
            freq = frequency_log.get((layer, expert), 0)
            candidates.append((freq, layer, expert))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    gpu_placements = {l: set() for l in range(num_layers)}
    used_vram = 0

    for freq, l, e in candidates:
        if used_vram + expert_bytes <= vram_budget:
            gpu_placements[l].add(e)
            used_vram += expert_bytes

    res = {}
    for l in range(num_layers):
        res[l] = {
            "gpu": sorted(list(gpu_placements[l])),
            "cpu": sorted([e for e in range(num_experts) if e not in gpu_placements[l]])
        }
    return res

def simulate_lru_cache(capacity, trace):
    if not trace or capacity <= 0:
        return 0.0
    cache = []
    hits = 0
    for item in trace:
        if item in cache:
            hits += 1
            cache.remove(item)
            cache.append(item)
        else:
            if len(cache) >= capacity:
                cache.pop(0)
            cache.append(item)
    return hits / len(trace)

def evaluate_offload_latency(num_layers, token_count, gpu_layer_time, cpu_layer_time, pcie_transfer_time, offload_first_n):
    n = max(0, min(num_layers, offload_first_n))
    cpu_layers = n
    gpu_layers = num_layers - n

    layer_time_all_offload = cpu_layer_time * num_layers
    total_offload_all = token_count * layer_time_all_offload

    layer_time_split = (cpu_layers * cpu_layer_time) + (gpu_layers * gpu_layer_time)
    transfer_penalty = pcie_transfer_time if (cpu_layers > 0 and gpu_layers > 0) else 0.0
    total_offload_split = token_count * (layer_time_split + transfer_penalty)

    return {
        "offload_all_latency": float(total_offload_all),
        "offload_split_latency": float(total_offload_split),
        "speedup": float(total_offload_all / total_offload_split) if total_offload_split > 0 else 1.0
    }
