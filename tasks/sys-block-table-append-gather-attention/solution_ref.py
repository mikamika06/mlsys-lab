import math
import numpy as np


def paged_append(k_pool, v_pool, block_table, free_blocks, new_k, new_v, block_size):
    """Append new tokens to a single sequence's PagedAttention KV cache,
    growing the block table by allocating fresh physical blocks on demand.

    k_pool, v_pool : (num_phys_blocks, block_size, D) float64 arrays -- the
        shared physical KV pool. Mutated in place.
    block_table : list[int]
        Physical block id backing each already-allocated logical block of
        this sequence (may be empty at the start). Mutated in place
        (grown by exactly as many blocks as needed).
    free_blocks : list[int]
        Available physical block ids. Treated as a sorted queue: the
        smallest available id is always allocated next
        (`free_blocks.pop(0)`), for a deterministic, reproducible
        allocation order. Mutated in place.
    new_k, new_v : (L, D) float64 arrays
        The L new tokens' key/value vectors to append.
    block_size : int

    Before this call, `n_before = len(block_table) * block_size` tokens of
    this sequence are already resident. New token `i` (0-indexed) is
    written to logical position `n_before + i`. Whenever that position's
    logical block index (`(n_before + i) // block_size`) does not yet
    exist in `block_table`, a fresh physical block is allocated (popped
    from `free_blocks`) and appended to `block_table` before writing.

    Returns
    -------
    seq_len : int
        The new total number of resident tokens, `n_before + L`.
    """
    new_k = np.asarray(new_k, dtype=np.float64)
    new_v = np.asarray(new_v, dtype=np.float64)
    n_before = len(block_table) * block_size
    L = new_k.shape[0]

    for i in range(L):
        pos = n_before + i
        logical_block = pos // block_size
        slot = pos % block_size
        if logical_block >= len(block_table):
            block_table.append(free_blocks.pop(0))
        phys = block_table[logical_block]
        k_pool[phys, slot] = new_k[i]
        v_pool[phys, slot] = new_v[i]

    return n_before + L


def gather_and_attend(k_pool, v_pool, block_table, block_size, seq_len, q):
    """Gather the logical KV sequence from the paged pool via the block
    table, then run single-query scaled dot-product attention.

    k_pool, v_pool : (num_phys_blocks, block_size, D)
    block_table    : list[int], length >= ceil(seq_len / block_size)
    seq_len        : int, number of valid logical tokens
    q              : (D,) float64 array

    Returns
    -------
    out : (D,) float64 array
        softmax(q @ K^T / sqrt(D)) @ V over the gathered K, V.
    """
    k_pool = np.asarray(k_pool, dtype=np.float64)
    v_pool = np.asarray(v_pool, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    D = q.shape[0]
    k_logical_list = []
    v_logical_list = []
    for i in range(seq_len):
        logical_block = i // block_size
        slot = i % block_size
        phys = block_table[logical_block]
        k_logical_list.append(k_pool[phys, slot])
        v_logical_list.append(v_pool[phys, slot])

    k_logical = np.asarray(k_logical_list, dtype=np.float64)
    v_logical = np.asarray(v_logical_list, dtype=np.float64)

    scores_list = []
    scale = math.sqrt(D)
    for i in range(seq_len):
        dot = 0.0
        for j in range(D):
            dot += k_logical[i, j] * q[j]
        scores_list.append(dot / scale)

    max_score = scores_list[0]
    for s in scores_list:
        if s > max_score:
            max_score = s

    w_list = []
    sum_w = 0.0
    for s in scores_list:
        val = math.exp(s - max_score)
        w_list.append(val)
        sum_w += val

    w_normalized = [val / sum_w for val in w_list]

    out = np.zeros(D, dtype=np.float64)
    for j in range(D):
        acc = 0.0
        for i in range(seq_len):
            acc += w_normalized[i] * v_logical[i, j]
        out[j] = acc

    return out
