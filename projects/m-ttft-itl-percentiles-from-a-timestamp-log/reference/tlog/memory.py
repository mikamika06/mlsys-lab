def calculate_kv_memory(num_layers, num_kv_heads, head_dim, max_seq_len, dtype_bytes, total_memory):
    bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes
    bytes_per_seq = bytes_per_token_per_layer * num_layers * max_seq_len
    max_seqs = int(total_memory // bytes_per_seq) if bytes_per_seq > 0 else 0
    return {
        "bytes_per_seq": int(bytes_per_seq),
        "max_sequences": max_seqs
    }
