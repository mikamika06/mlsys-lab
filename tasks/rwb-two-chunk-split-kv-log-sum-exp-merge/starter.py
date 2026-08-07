import math

def two_chunk_split_kv_merge(q, k, v):
    """
    q: (d,) single decode query.
    k, v: (N, d) and (N, d_v) full KV cache for this query.

    Split k/v into two contiguous chunks at the midpoint (N // 2). For
    each chunk compute its local normalized attention output O_i and its
    true log-sum-exp L_i (from that chunk's 1/sqrt(d)-scaled scores),
    then merge:

        m = max(L1, L2)
        O = (O1 * exp(L1-m) + O2 * exp(L2-m)) / (exp(L1-m) + exp(L2-m))

    Returns the merged (d_v,) float64 output, equal to full single-pass
    attention over the whole (unsplit) KV cache.
    """
    raise NotImplementedError('your code here')
