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
    raise NotImplementedError('your code here')


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
    raise NotImplementedError('your code here')
