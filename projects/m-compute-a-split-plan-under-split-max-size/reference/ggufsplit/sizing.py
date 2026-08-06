def predict_shard_sizes(shards):
    return [sum(size for _, size in shard) for shard in shards]
