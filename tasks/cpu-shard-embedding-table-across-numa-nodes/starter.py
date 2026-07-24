def shard_embedding_table(num_embeddings: int, dim: int, num_nodes: int) -> list[tuple[int, int]]:
    """Return contiguous row ranges assigning embedding table rows to each NUMA node."""
    raise NotImplementedError("implement equal contiguous partition")
