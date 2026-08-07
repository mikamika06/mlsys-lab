from .memory import compute_vram_saved

def find_min_cpu_moe(tensors, vram_budget):
    moe_layers = set()
    for t_info in tensors.values():
        if t_info.get("is_moe", False):
            moe_layers.add(t_info.get("moe_layer", -1))

    max_layer = max(moe_layers) if moe_layers else 0
    total_moe_layers = max_layer + 1 if moe_layers else 0

    total_vram = sum(t["size_bytes"] for t in tensors.values())

    for n in range(total_moe_layers + 1):
        saved = compute_vram_saved(tensors, n)
        if (total_vram - saved) <= vram_budget:
            return n

    return total_moe_layers
