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
