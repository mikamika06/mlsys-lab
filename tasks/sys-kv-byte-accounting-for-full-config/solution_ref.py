import numpy as np

def kv_cache_bytes(layers, heads, d_kv, seq_len, dtype, batch):
    """
    Compute the total number of bytes required to store a full KV cache.

    Parameters
    ----------
    layers : int
        Number of transformer layers.
    heads : int
        Number of attention heads per layer.
    d_kv : int
        Dimensionality of each key/value vector (assumed equal).
    seq_len : int
        Maximum sequence length that will be cached.
    dtype : np.dtype or type
        NumPy data type used for the tensors.
    batch : int
        Batch size.

    Returns
    -------
    int
        Total byte count.
    """
    return int(
        layers
        * heads
        * 2          # key + value per head
        * batch
        * seq_len
        * d_kv
        * np.dtype(dtype).itemsize
    )
