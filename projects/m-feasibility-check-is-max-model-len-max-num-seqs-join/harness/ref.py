import math

def compute_kv_bytes(max_model_len, max_num_seqs, num_layers, kv_heads, head_dim, block_size, dtype_bytes):
    total_tokens = max_model_len * max_num_seqs
    num_blocks = math.ceil(total_tokens / block_size)
    bytes_per_block = block_size * num_layers * kv_heads * head_dim * 2 * dtype_bytes
    return num_blocks * bytes_per_block

def is_feasible(max_model_len, max_num_seqs, gpu_memory_bytes, num_layers, kv_heads, head_dim, block_size, dtype_bytes, overhead_bytes=0):
    needed = compute_kv_bytes(max_model_len, max_num_seqs, num_layers, kv_heads, head_dim, block_size, dtype_bytes)
    return (needed + overhead_bytes) <= gpu_memory_bytes

def quant_gain_and_risk(fp16_bytes, fp8_bytes):
    gain = fp16_bytes / fp8_bytes
    risk = "quantization error"
    return gain, risk

def repair_launch(max_model_len, max_num_seqs, gpu_memory_bytes, num_layers, kv_heads, head_dim, block_size, dtype_bytes, overhead_bytes=0):
    if is_feasible(max_model_len, max_num_seqs, gpu_memory_bytes, num_layers, kv_heads, head_dim, block_size, dtype_bytes, overhead_bytes):
        return max_model_len, max_num_seqs
    needed = compute_kv_bytes(max_model_len, max_num_seqs, num_layers, kv_heads, head_dim, block_size, dtype_bytes) + overhead_bytes
    ratio = gpu_memory_bytes / needed
    new_num_seqs = max(1, int(max_num_seqs * ratio * 0.95))
    new_model_len = max(512, int(max_model_len * ratio * 0.95))
    return new_model_len, new_num_seqs

CONFIGS = [
    {"max_model_len": 4096, "max_num_seqs": 256, "gpu_memory_bytes": 16 * 1024**3, "num_layers": 32, "kv_heads": 8, "head_dim": 128, "block_size": 16, "dtype_bytes": 2, "overhead_bytes": 2 * 1024**3},
    {"max_model_len": 8192, "max_num_seqs": 512, "gpu_memory_bytes": 24 * 1024**3, "num_layers": 40, "kv_heads": 8, "head_dim": 128, "block_size": 16, "dtype_bytes": 2, "overhead_bytes": 2 * 1024**3},
    {"max_model_len": 16384, "max_num_seqs": 128, "gpu_memory_bytes": 40 * 1024**3, "num_layers": 80, "kv_heads": 8, "head_dim": 128, "block_size": 16, "dtype_bytes": 2, "overhead_bytes": 4 * 1024**3},
    {"max_model_len": 2048, "max_num_seqs": 1024, "gpu_memory_bytes": 8 * 1024**3, "num_layers": 24, "kv_heads": 4, "head_dim": 128, "block_size": 16, "dtype_bytes": 2, "overhead_bytes": 1 * 1024**3},
    {"max_model_len": 32768, "max_num_seqs": 64, "gpu_memory_bytes": 80 * 1024**3, "num_layers": 64, "kv_heads": 8, "head_dim": 128, "block_size": 16, "dtype_bytes": 2, "overhead_bytes": 5 * 1024**3},
]

QUANT_CONFIGS = [
    {"fp16_bytes": 2.0, "fp8_bytes": 1.0},
    {"fp16_bytes": 2.0, "fp8_bytes": 1.0},
    {"fp16_bytes": 4.0, "fp8_bytes": 2.0},
    {"fp16_bytes": 2.0, "fp8_bytes": 1.0},
    {"fp16_bytes": 2.0, "fp8_bytes": 1.0},
]
