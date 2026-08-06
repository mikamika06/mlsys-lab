from actmem.formula import compute_layer_activation_bytes

def compute_total_uncheckpointed_memory(num_layers: int, b: int, s: int, h: int, heads: int, intermediate_size: int, dtype_bytes: int) -> int:
    layer_bytes = compute_layer_activation_bytes(b, s, h, heads, dtype_bytes)
    mlp_extra = b * s * intermediate_size * dtype_bytes
    return num_layers * (layer_bytes + mlp_extra)
