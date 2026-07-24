import numpy as np

def kv_peak_bytes(num_layers: int, batch_size: int, num_heads: int, seq_len: int, head_dim: int, dtype: np.dtype) -> tuple[float, float]:
    """TODO: This implementation incorrectly assumes that the offloaded
strategy keeps all layers resident and that the full cache strategy
only keeps two layers.  It will therefore produce swapped values."""
    raise NotImplementedError('your code here')
