import math

def check_feasibility(max_model_len, max_num_seqs, gpu_memory_bytes, num_layers, kv_heads, head_dim, block_size, dtype_bytes, overhead_bytes=0):
    total_tokens = max_model_len * max_num_seqs
    num_blocks = math.ceil(total_tokens / block_size)
    bytes_per_block = block_size * num_layers * kv_heads * head_dim * 2 * dtype_bytes
    needed = num_blocks * bytes_per_block + overhead_bytes
    return needed <= gpu_memory_bytes
