def dtype_size(dtype_str):
    raise NotImplementedError


def calculate_eager_memory(batch_size, seq_len, num_heads, head_dim, dtype_str="float16"):
    raise NotImplementedError


def calculate_sdpa_memory(batch_size, seq_len, num_heads, head_dim, dtype_str="float16"):
    raise NotImplementedError
