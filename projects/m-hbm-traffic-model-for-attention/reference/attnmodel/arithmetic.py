from attnmodel.traffic import hbm_traffic

def compute_flops(batch_size, seq_len, num_heads, head_dim):
    return 4 * batch_size * num_heads * seq_len * seq_len * head_dim

def arithmetic_intensity(batch_size, seq_len, num_heads, head_dim, bytes_per_elem=2):
    flops = compute_flops(batch_size, seq_len, num_heads, head_dim)
    traffic = hbm_traffic(batch_size, seq_len, num_heads, head_dim, bytes_per_elem)
    return flops / traffic
