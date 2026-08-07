import math


def paged_attention(q: list[float], k_pool: list[list[list[float]]], v_pool: list[list[list[float]]], block_table: list[int], seq_len: int, block_size: int) -> list[float]:
    """Single-query attention over a KV cache stored in a paged physical pool.

    q            : (d,) query vector for the newly generated token.
    k_pool,v_pool: (num_physical_blocks, block_size, d) SHARED physical pool.
        May hold other requests' data (or stale garbage) outside the blocks
        this request owns, and past `seq_len` inside its last logical block.
    block_table  : list of ints, length ceil(seq_len / block_size).
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
    Returns a list of length d.
    """
    d = len(q)

    K = []
    V = []
    for pos in range(seq_len):
        logical_block = pos // block_size
        slot = pos % block_size
        phys = int(block_table[logical_block])
        K.append(k_pool[phys][slot])
        V.append(v_pool[phys][slot])

    inv_sqrt_d = 1.0 / math.sqrt(d)
    scores = []
    for i in range(seq_len):
        dot = 0.0
        for j in range(d):
            dot += K[i][j] * q[j]
        scores.append(dot * inv_sqrt_d)

    max_score = scores[0]
    for i in range(1, seq_len):
        if scores[i] > max_score:
            max_score = scores[i]

    probs = []
    sum_exp = 0.0
    for i in range(seq_len):
        val = math.exp(scores[i] - max_score)
        probs.append(val)
        sum_exp += val

    for i in range(seq_len):
        probs[i] /= sum_exp

    out = [0.0] * d
    for j in range(d):
        acc = 0.0
        for i in range(seq_len):
            acc += probs[i] * V[i][j]
        out[j] = acc

    return out
