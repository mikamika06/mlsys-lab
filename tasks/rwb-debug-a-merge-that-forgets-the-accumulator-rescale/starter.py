import math

def merge_split_kv(partials: list[tuple[float, float, list[float]]]) -> list[float]:
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
    raise NotImplementedError('your code here')
