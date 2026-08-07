from moeoff.vram import compute_vram_saved

def find_min_cpu_moe(tensors, expert_tensor_patterns, total_layers, max_vram_bytes):
    total_size = sum(size for _, size in tensors)
    for n in range(total_layers + 1):
        saved = compute_vram_saved(tensors, expert_tensor_patterns, n, total_layers)
        current_vram = total_size - saved
        if current_vram <= max_vram_bytes:
            return n
    return total_layers
