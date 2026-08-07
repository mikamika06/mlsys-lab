import math


def merge_split_kv(partials: list[tuple[float, float, list[float]]]) -> list[float]:
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
    m = max(p[0] for p in partials)

    l = 0.0
    numerator = None
    for m_i, l_i, o_i in partials:
        scale = math.exp(m_i - m)
        l += scale * l_i
        term = [scale * v_j for v_j in o_i]
        numerator = term if numerator is None else [n + t for n, t in zip(numerator, term)]

    return [n / l for n in numerator]
