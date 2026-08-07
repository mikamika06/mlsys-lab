import re

def compute_vram_saved(tensors, expert_tensor_patterns, n_cpu_moe, total_layers):
    saved = 0
    start_layer = total_layers - n_cpu_moe
    for name, size in tensors:
        m = re.search(r"blk\.(\d+)\.", name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx >= start_layer:
                for pat in expert_tensor_patterns:
                    if re.search(pat, name):
                        saved += size
                        break
    return saved
