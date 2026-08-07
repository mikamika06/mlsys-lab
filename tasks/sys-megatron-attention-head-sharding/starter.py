import math

def sharded_attention_heads(q: list[list[list[list[float]]]], k: list[list[list[list[float]]]], v: list[list[list[list[float]]]], wo: list[list[float]], num_ranks: int) -> list[list[list[float]]]:
    """Compute tensor-parallel attention head sharding."""
    raise NotImplementedError('your code here')
