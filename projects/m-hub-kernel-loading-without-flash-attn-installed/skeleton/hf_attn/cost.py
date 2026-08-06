def estimate_train_step_cost(batch_size, seq_len, num_heads, head_dim, backend="sdpa"):
    raise NotImplementedError


def compare_backend_costs(batch_size, seq_len, num_heads, head_dim, available_backends):
    raise NotImplementedError
