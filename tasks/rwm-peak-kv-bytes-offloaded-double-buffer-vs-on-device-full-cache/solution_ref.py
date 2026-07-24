import numpy as np

def kv_peak_bytes(num_layers: int,
                  batch_size: int,
                  num_heads: int,
                  seq_len: int,
                  head_dim: int,
                  dtype: np.dtype) -> tuple[float, float]:
    """
    Compute the peak resident KV bytes for two strategies:

    - Offloaded double buffer: only two layers are resident.
    - On‑device full cache: all layers are resident.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers (L).
    batch_size : int
        Batch size (B).
    num_heads : int
        Number of attention heads (H).
    seq_len : int
        Sequence length (S).
    head_dim : int
        Head dimension (D).
    dtype : np.dtype
        Data type of the KV tensors.

    Returns
    -------
    tuple[float, float]
        (peak_offloaded_bytes, peak_full_cache_bytes)
    """
    b = dtype.itemsize  # bytes per element
    layer_kv = 2 * batch_size * num_heads * seq_len * head_dim * b
    peak_offloaded = 2 * layer_kv          # double buffer across layers
    peak_full_cache = num_layers * layer_kv  # all layers resident
    return float(peak_offloaded), float(peak_full_cache)
