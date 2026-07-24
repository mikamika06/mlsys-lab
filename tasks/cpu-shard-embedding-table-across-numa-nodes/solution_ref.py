def shard_embedding_table(num_embeddings: int, dim: int, num_nodes: int) -> list[tuple[int, int]]:
    base = num_embeddings // num_nodes
    rem = num_embeddings % num_nodes
    shards = []
    start = 0
    for i in range(num_nodes):
        extra = 1 if i < rem else 0
        end = start + base + extra
        shards.append((start, end))
        start = end
    return shards
