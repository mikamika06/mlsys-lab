import numpy as np


def paged_append_and_attend(kv_pool_k: np.ndarray, kv_pool_v: np.ndarray,
                             block_table: list[int], block_size: int, existing_len: int,
                             new_k: np.ndarray, new_v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """PagedAttention-style append: write new tokens' K/V into a paged
    physical pool via the slot mapping, then gather the full sequence
    back out through the same mapping and attend.

    kv_pool_k, kv_pool_v : (num_physical_blocks * block_size, d) flat
        physical KV pool, shared across sequences. Rows for this
        sequence's positions 0..existing_len-1 are ALREADY written at
        their correct physical slots (see slot formula below).
    block_table : list of physical block ids for this sequence's LOGICAL
        blocks 0, 1, 2, ... in order. Physical block ids need not be
        contiguous or in logical order.
    block_size  : tokens per block.
    existing_len: number of tokens already written for this sequence.
    new_k, new_v: (T, d) new keys/values for positions
        existing_len .. existing_len+T-1. May span more than one block.
    q : (d,) query attended AFTER the append, over all existing_len + T
        tokens.

    The physical slot for absolute position `pos` is:
        logical_block = pos // block_size
        offset        = pos %  block_size
        slot          = block_table[logical_block] * block_size + offset

    Returns the (d,) scaled dot-product attention output of `q` over the
    full existing_len + T tokens, gathered from the pool (not assumed
    contiguous).
    """
    kv_pool_k = np.array(kv_pool_k, dtype=np.float64, copy=True)
    kv_pool_v = np.array(kv_pool_v, dtype=np.float64, copy=True)
    new_k = np.asarray(new_k, dtype=np.float64)
    new_v = np.asarray(new_v, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    T, d = new_k.shape

    def slot_of(pos: int) -> int:
        logical_block = pos // block_size
        offset = pos % block_size
        return block_table[logical_block] * block_size + offset

    # write path: append new_k/new_v into the physical pool via slot_mapping
    for i in range(T):
        pos = existing_len + i
        s = slot_of(pos)
        kv_pool_k[s] = new_k[i]
        kv_pool_v[s] = new_v[i]

    # gather path: reassemble the logical sequence from the physical pool
    total_len = existing_len + T
    gathered_k = np.empty((total_len, d), dtype=np.float64)
    gathered_v = np.empty((total_len, d), dtype=np.float64)
    for pos in range(total_len):
        s = slot_of(pos)
        gathered_k[pos] = kv_pool_k[s]
        gathered_v[pos] = kv_pool_v[s]

    scores = (q @ gathered_k.T) / np.sqrt(d)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    return probs @ gathered_v
