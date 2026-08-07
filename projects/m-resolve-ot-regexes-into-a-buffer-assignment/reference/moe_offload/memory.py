def compute_vram_saved(tensors, n_cpu_moe):
    total_saved = 0
    for t_name, t_info in tensors.items():
        if t_info.get("is_moe", False):
            layer_idx = t_info.get("moe_layer", -1)
            if layer_idx < n_cpu_moe:
                total_saved += t_info["size_bytes"]
    return total_saved
