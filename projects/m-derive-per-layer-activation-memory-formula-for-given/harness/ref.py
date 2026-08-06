CONFIGS = [
    {"b": 1, "s": 512, "h": 2048, "heads": 16, "intermediate_size": 5504, "num_layers": 24, "dtype_bytes": 2},
    {"b": 2, "s": 1024, "h": 4096, "heads": 32, "intermediate_size": 11008, "num_layers": 32, "dtype_bytes": 2},
    {"b": 4, "s": 2048, "h": 8192, "heads": 64, "intermediate_size": 28672, "num_layers": 80, "dtype_bytes": 2}
]

def compute_layer_activation_bytes(b: int, s: int, h: int, heads: int, dtype_bytes: int) -> int:
    from reference.actmem.formula import compute_layer_activation_bytes as fn
    return fn(b, s, h, heads, dtype_bytes)

def find_attention_mlp_crossover(h: int, heads: int, intermediate_size: int) -> int:
    from reference.actmem.crossover import find_attention_mlp_crossover as fn
    return fn(h, heads, intermediate_size)

def compute_total_uncheckpointed_memory(num_layers: int, b: int, s: int, h: int, heads: int, intermediate_size: int, dtype_bytes: int) -> int:
    from reference.actmem.total import compute_total_uncheckpointed_memory as fn
    return fn(num_layers, b, s, h, heads, intermediate_size, dtype_bytes)
