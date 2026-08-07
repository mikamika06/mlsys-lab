def model_reduction_cost(batch_size, num_heads, kv_len, block_size, head_dim, num_sms, split_count, t_flop=1e-12, t_bw=1e-11, t_red=1e-10):
    raise NotImplementedError


def find_optimal_splits(batch_size, num_heads, kv_len, block_size, head_dim, num_sms, max_splits=128, t_flop=1e-12, t_bw=1e-11, t_red=1e-10):
    raise NotImplementedError
