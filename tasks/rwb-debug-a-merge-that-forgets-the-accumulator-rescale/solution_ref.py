import math
import numpy as np


def merge_split_kv(partials):
    """
    partials: list of (m_i, l_i, o_i) triples, one per KV shard, where
    (for that shard's logits over its own keys) m_i is the local max
    logit, l_i = sum(exp(logits - m_i)), and o_i = sum(exp(logits - m_i)
    * values). Combine them into the exact full-sequence softmax-attention
    output.

    The correction this fixes: each partial's numerator/denominator were
    scaled relative to its OWN local max m_i, not the true global max
    across all shards. Before summing, every partial must be RESCALED by
    exp(m_i - m) where m = max_i(m_i) is the global max, so that all
    partials are expressed on the same (numerically stable) scale before
    being combined.
    """
    m = partials[0][0]
    for p in partials:
        if p[0] > m:
            m = p[0]

    l = 0.0
    numerator_list = None
    arr_shape = None
    for m_i, l_i, o_i in partials:
        scale = math.exp(m_i - m)
        l += scale * l_i
        arr = np.asarray(o_i, dtype=np.float64)
        if arr_shape is None:
            arr_shape = arr.shape
        flat_arr = arr.ravel()
        if numerator_list is None:
            numerator_list = [0.0] * len(flat_arr)
            for idx in range(len(flat_arr)):
                numerator_list[idx] = scale * flat_arr[idx]
        else:
            for idx in range(len(flat_arr)):
                numerator_list[idx] += scale * flat_arr[idx]

    final_list = [val / l for val in numerator_list]
    return np.asarray(final_list, dtype=np.float64).reshape(arr_shape)
