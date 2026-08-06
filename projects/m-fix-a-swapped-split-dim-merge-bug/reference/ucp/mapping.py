def reconstruct_mapping(config):
    name = config["param_name"]
    shape = config["shape"]
    shards = config["shards"]
    split_dim = config["split_dim"]
    shard_shape = list(shape)
    shard_shape[split_dim] = shape[split_dim] // shards
    res = []
    for i in range(shards):
        res.append({
            "shard_id": i,
            "name": f"{name}.shard_{i}",
            "shape": tuple(shard_shape),
            "split_dim": split_dim
        })
    return res
