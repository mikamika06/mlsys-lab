import numpy as np


def paged_attention(q, k_pool, v_pool, block_table, seq_len, block_size):
    """Single-query attention over a KV cache stored in a paged physical pool.

    q            : (d,) query vector for the newly generated token.
    k_pool,v_pool: (num_physical_blocks, block_size, d) SHARED physical pool.
        May hold other requests' data (or stale garbage) outside the blocks
        this request owns, and past `seq_len` inside its last logical block.
    block_table  : 1-D int array, length ceil(seq_len / block_size).
        block_table[b] is the physical block backing this request's logical
        block b.
    seq_len      : number of VALID cached tokens for this request.
    block_size   : token slots per physical block.

    Gather exactly the `seq_len` valid K/V vectors (never read past
    seq_len, even within an otherwise-valid block), then compute standard
    scaled dot-product attention of q against them (no causal mask needed --
    every cached token already precedes the query):
        probs = softmax(K @ q / sqrt(d))
        out   = probs @ V
    Returns a (d,) vector.
    """
    q = np.asarray(q, dtype=np.float64)
    d = q.shape[0]

    K = np.empty((seq_len, d), dtype=np.float64)
    V = np.empty((seq_len, d), dtype=np.float64)
    for pos in range(seq_len):
        logical_block = pos // block_size
        slot = pos % block_size
        phys = int(block_table[logical_block])
        K[pos] = k_pool[phys, slot]
        V[pos] = v_pool[phys, slot]

    scores = (K @ q) / np.sqrt(d)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    return probs @ V
