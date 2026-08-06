import math

def repair_launch(max_model_len, max_num_seqs, gpu_memory_bytes, num_layers, kv_heads, head_dim, block_size, dtype_bytes, overhead_bytes=0):
    total_tokens = max_model_len * max_num_seqs
    num_blocks = math.ceil(total_tokens / block_size)
    bytes_per_block = block_size * num_layers * kv_heads * head_dim * 2 * dtype_bytes
    needed = num_blocks * bytes_per_block + overhead_bytes
    if needed <= gpu_memory_bytes:
        return max_model_len, max_num_seqs
    ratio = gpu_memory_bytes / needed
    new_num_seqs = max(1, int(max_num_seqs * ratio * 0.95))
    new_model_len = max(512, int(max_model_len * ratio * 0.95))
    return new_model_len, new_num_seqs
