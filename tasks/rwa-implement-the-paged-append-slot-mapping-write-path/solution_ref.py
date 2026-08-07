import math


def paged_append_and_attend(
    kv_pool_k: list[list[float]],
    kv_pool_v: list[list[float]],
    block_table: list[int],
    block_size: int,
    existing_len: int,
    new_k: list[list[float]],
    new_v: list[list[float]],
    q: list[float],
) -> list[float]:
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
    T = len(new_k)
    d = len(q)

    def slot_of(pos: int) -> int:
        logical_block = pos // block_size
        offset = pos % block_size
        return block_table[logical_block] * block_size + offset

    for i in range(T):
        pos = existing_len + i
        s = slot_of(pos)
        for j in range(d):
            kv_pool_k[s][j] = new_k[i][j]
            kv_pool_v[s][j] = new_v[i][j]

    total_len = existing_len + T
    gathered_k = [[0.0] * d for _ in range(total_len)]
    gathered_v = [[0.0] * d for _ in range(total_len)]
    for pos in range(total_len):
        s = slot_of(pos)
        for j in range(d):
            gathered_k[pos][j] = kv_pool_k[s][j]
            gathered_v[pos][j] = kv_pool_v[s][j]

    scores = [0.0] * total_len
    scale = math.sqrt(d)
    for i in range(total_len):
        dot = 0.0
        for j in range(d):
            dot += q[j] * gathered_k[i][j]
        scores[i] = dot / scale

    max_score = scores[0]
    for i in range(1, total_len):
        if scores[i] > max_score:
            max_score = scores[i]

    probs = [0.0] * total_len
    sum_probs = 0.0
    for i in range(total_len):
        val = math.exp(scores[i] - max_score)
        probs[i] = val
        sum_probs += val

    for i in range(total_len):
        probs[i] /= sum_probs

    out = [0.0] * d
    for j in range(d):
        acc = 0.0
        for i in range(total_len):
            acc += probs[i] * gathered_v[i][j]
        out[j] = acc

    return out
