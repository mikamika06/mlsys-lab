def hbm_traffic(batch_size, seq_len, num_heads, head_dim, bytes_per_elem=2):
    qkv = 3 * batch_size * num_heads * seq_len * head_dim * bytes_per_elem
    matrix = batch_size * num_heads * seq_len * seq_len * bytes_per_elem
    out = batch_size * num_heads * seq_len * head_dim * bytes_per_elem
    return qkv + 3 * matrix + out
