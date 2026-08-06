import numpy as np


class KVCacheBuffer:
    """Preallocated KV cache buffer indexed by physical cache batch slots."""

    def __init__(self, max_cache_batch, max_seq_len, num_heads, head_dim, dtype=np.float32):
        self.max_cache_batch = max_cache_batch
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k = np.zeros((max_cache_batch, max_seq_len, num_heads, head_dim), dtype=dtype)
        self.v = np.zeros((max_cache_batch, max_seq_len, num_heads, head_dim), dtype=dtype)

    def update_and_fetch(self, cache_batch_idx, seq_lens, new_k, new_v):
        """Insert new token K/V at seq_lens position and gather current sequence history."""
        cache_batch_idx = np.asarray(cache_batch_idx, dtype=np.int32)
        seq_lens = np.asarray(seq_lens, dtype=np.int32)
        batch_size = len(cache_batch_idx)

        for i in range(batch_size):
            c_idx = cache_batch_idx[i]
            pos = seq_lens[i]
            if pos >= self.max_seq_len:
                raise ValueError(f"Sequence length {pos} exceeds max_seq_len {self.max_seq_len}")
            self.k[c_idx, pos] = new_k[i]
            self.v[c_idx, pos] = new_v[i]

        updated_lens = seq_lens + 1
        max_len = int(np.max(updated_lens)) if batch_size > 0 else 0
        k_out = np.zeros((batch_size, max_len, self.num_heads, self.head_dim), dtype=self.k.dtype)
        v_out = np.zeros((batch_size, max_len, self.num_heads, self.head_dim), dtype=self.v.dtype)

        for i in range(batch_size):
            c_idx = cache_batch_idx[i]
            length = updated_lens[i]
            k_out[i, :length] = self.k[c_idx, :length]
            v_out[i, :length] = self.v[c_idx, :length]

        return k_out, v_out
