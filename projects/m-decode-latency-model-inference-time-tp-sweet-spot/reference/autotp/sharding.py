def shard_kv_heads(num_kv_heads, tp_degree):
    base = num_kv_heads // tp_degree
    rem = num_kv_heads % tp_degree
    shards = []
    for i in range(tp_degree):
        shards.append(base + (1 if i < rem else 0))
    return shards
