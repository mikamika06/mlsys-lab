import numpy as np


def gather_batch_kv(kv_cache, cache_batch_idx, seq_lens):
    """Gather sequence key and value history from cache buffer using physical batch indices."""
    cache_batch_idx = np.asarray(cache_batch_idx, dtype=np.int32)
    seq_lens = np.asarray(seq_lens, dtype=np.int32)
    batch_size = len(cache_batch_idx)

    max_len = int(np.max(seq_lens)) if batch_size > 0 else 0
    k_batch = np.zeros((batch_size, max_len, kv_cache.num_heads, kv_cache.head_dim), dtype=kv_cache.k.dtype)
    v_batch = np.zeros((batch_size, max_len, kv_cache.num_heads, kv_cache.head_dim), dtype=kv_cache.v.dtype)

    for i in range(batch_size):
        c_idx = cache_batch_idx[i]
        length = seq_lens[i]
        if length > 0:
            k_batch[i, :length] = kv_cache.k[c_idx, :length]
            v_batch[i, :length] = kv_cache.v[c_idx, :length]

    return k_batch, v_batch
