import numpy as np


def merge_split_kv(partials):
    """
    partials: list of (m_i, l_i, o_i) triples, one per KV shard, where
    (for that shard's logits over its own keys) m_i is the local max
    logit, l_i = sum(exp(logits - m_i)), and o_i = sum(exp(logits - m_i)
    * values). Combine them into the full-sequence softmax-attention
    output.

    BUG: this weights each partial by exp(m_i) directly and never
    computes (let alone rescales by) the global max m = max_i(m_i) across
    shards. Fix it.
    """
    l = 0.0
    numerator = None
    for m_i, l_i, o_i in partials:
        scale = np.exp(m_i)
        l += scale * l_i
        term = scale * np.asarray(o_i, dtype=np.float64)
        numerator = term if numerator is None else numerator + term

    return numerator / l
