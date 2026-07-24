import numpy as np

def kv_cache_bytes(layers: int,
                   heads: int,
                   head_dim: int,
                   seq_len: int,
                   dtype: str) -> int:
    """
    Return the number of bytes required to store a KV cache for a transformer
    model with the given configuration.

    Parameters
    ----------
    layers : int
        Number of decoder layers.
    heads : int
        Attention heads per layer.
    head_dim : int
        Dimension of each head (key/value vector size).
    seq_len : int
        Current context length (number of tokens decoded).
    dtype : str or numpy.dtype
        Data type used to store keys and values.

    Returns
    -------
    int
        Total byte count for the KV cache.
    """
    dt = np.dtype(dtype)
    return layers * heads * seq_len * head_dim * 2 * dt.itemsize
