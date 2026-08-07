def append_paged_kv(k_pool, v_pool, new_k, new_v, cu_new_seqlens, seq_start_pos, block_tables, block_size):
    """Write each request's newly computed K/V vectors into their correct
    physical slots in a paged KV cache pool (PagedAttention-style append).

    k_pool, v_pool : lists of lists of lists of floats --
        the SHARED physical KV pool. Mutated IN PLACE.
    new_k, new_v   : lists of lists of floats -- new K/V vectors
        for ALL requests in the batch, packed back-to-back (ragged).
    cu_new_seqlens : list of ints, length num_requests + 1. Request r's new
        tokens are new_k[cu_new_seqlens[r]:cu_new_seqlens[r+1]].
    seq_start_pos  : list of ints, length num_requests. seq_start_pos[r] is
        the number of tokens ALREADY in request r's cache before this call
        (the logical sequence position of the first newly appended token).
    block_tables   : list of lists of ints, length num_requests.
        block_tables[r][b] is the PHYSICAL block index backing request r's
        LOGICAL block b (each logical block holds `block_size` token slots).

    For request r and its i-th new token (0-indexed), the ABSOLUTE logical
    position is pos = seq_start_pos[r] + i. That token is written to
        physical block  = block_tables[r][pos // block_size]
        slot in block   = pos % block_size

    Returns None; k_pool and v_pool are modified in place.
    """
    num_requests = len(cu_new_seqlens) - 1
    for r in range(num_requests):
        start = int(cu_new_seqlens[r])
        end = int(cu_new_seqlens[r + 1])
        base_pos = int(seq_start_pos[r])
        bt = block_tables[r]
        for i in range(end - start):
            pos = base_pos + i
            logical_block = pos // block_size
            slot = pos % block_size
            phys = int(bt[logical_block])
            k_pool[phys][slot] = list(new_k[start + i])
            v_pool[phys][slot] = list(new_v[start + i])
    return None
