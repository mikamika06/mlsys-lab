import numpy as np


class KVCacheBuffer:
    """Preallocated KV cache buffer indexed by physical cache batch slots."""

    def __init__(self, max_cache_batch, max_seq_len, num_heads, head_dim, dtype=np.float32):
        raise NotImplementedError

    def update_and_fetch(self, cache_batch_idx, seq_lens, new_k, new_v):
        """Insert new token K/V at seq_lens position and gather current sequence history."""
        raise NotImplementedError
