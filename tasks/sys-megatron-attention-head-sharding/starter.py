import numpy as np


def sharded_attention_heads(q, k, v, wo, num_ranks):
    """Compute tensor-parallel attention head sharding."""
    raise NotImplementedError('your code here')
