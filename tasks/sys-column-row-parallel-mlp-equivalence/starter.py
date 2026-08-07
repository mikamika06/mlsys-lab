import math

def mlp_tensor_parallel(x: list[list[float]], w1_shards: list[list[list[float]]], b1_shards: list[list[float]], w2_shards: list[list[list[float]]], b2: list[float]) -> list[list[float]]:
    """Column-parallel then row-parallel tensor-parallel MLP forward pass.

    x: (m, d) input.
    w1_shards: list of (d, h_i) first-layer weight shards.
    b1_shards: list of (h_i,) first-layer bias shards.
    w2_shards: list of (h_i, d_out) second-layer weight shards.
    b2: (d_out,) replicated output bias, added once.
    Returns the combined MLP output as a float64 array of shape (m, d_out).
    """
    raise NotImplementedError('your code here')
