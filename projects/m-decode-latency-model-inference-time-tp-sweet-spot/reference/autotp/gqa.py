def shard_kv_heads(num_kv_heads, tp_degree):
    if num_kv_heads % tp_degree == 0:
        return num_kv_heads // tp_degree
    elif tp_degree % num_kv_heads == 0:
        return 1
    else:
        return max(1, num_kv_heads // tp_degree)
