import numpy as np

ERROR_MESSAGES = [
    "CUDA error: invalid argument at block_dim=(128, 4, 1) tensor_stride=512 property=layout_stride",
    "RuntimeError: FlashAttention kernel failed: mismatch in head_dim expected 64 got 128 property=head_dim",
    "AssertionError: kernel launch failed due to unaligned pointer property=alignment_offset",
    "CUDA kernel error: sequence length exceeds maximum supported tile property=seq_len_limit",
]

def parse_error_property(msg):
    parts = msg.split("property=")
    if len(parts) > 1:
        return {"property": parts[1].strip(), "raw": msg}
    return {"property": "unknown", "raw": msg}

STATES = [
    {"layer.0.weight": "float16", "layer.0.lora_A": "float32", "layer.0.lora_B": "float16"},
    {"layer.1.weight": "bfloat16", "layer.1.lora_A": "bfloat16", "layer.1.lora_B": "bfloat16"},
    {"layer.2.weight": "float32", "layer.2.lora_A": "float16", "layer.2.lora_B": "float32"},
]

def find_dtype_leaks(state):
    leaks = []
    base_dtype = state.get("layer.0.weight", "float16")
    for k, v in state.items():
        if "lora" in k and v != base_dtype:
            leaks.append(k)
    return sorted(leaks)

def predict_max_batch(gpu_memory_bytes, seq_len, num_heads, head_dim):
    bytes_per_elem = 2
    attn_matrix_bytes = seq_len * seq_len * bytes_per_elem
    activations_per_layer = num_heads * head_dim * seq_len * bytes_per_elem * 4
    total_per_batch = attn_matrix_bytes + activations_per_layer
    if total_per_batch <= 0:
        return 1
    safe_memory = gpu_memory_bytes * 0.8
    max_b = int(safe_memory // total_per_batch)
    return max(1, max_b)
