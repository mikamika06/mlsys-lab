import numpy as np


def append_paged_kv(k_pool, v_pool, new_k, new_v, cu_new_seqlens, seq_start_pos, block_tables, block_size):
    """Write each request's newly computed K/V vectors into their correct
    physical slots in a paged KV cache pool (PagedAttention-style append).

    k_pool, v_pool : (num_physical_blocks, block_size, d) float64 arrays --
        the SHARED physical KV pool. Must be mutated IN PLACE.
    new_k, new_v   : (total_new_tokens, d) float64 arrays -- new K/V vectors
        for ALL requests in the batch, packed back-to-back (ragged).
    cu_new_seqlens : 1-D int array, length num_requests + 1.
    seq_start_pos  : 1-D int array, length num_requests -- tokens already in
        each request's cache before this call.
    block_tables   : list of 1-D int arrays, length num_requests -- logical
        block index -> physical block index per request.
    block_size     : token slots per physical block.

    See task.md for the exact slot-addressing formula.
    """
    raise NotImplementedError('your code here')
