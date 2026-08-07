import math

def paged_attention(q: list[float], k_pool: list[list[list[float]]], v_pool: list[list[list[float]]], block_table: list[int], seq_len: int, block_size: int) -> list[float]:
    """Single-query attention over a KV cache stored in a paged physical pool.

    q            : (d,) query vector for the newly generated token.
    k_pool,v_pool: (num_physical_blocks, block_size, d) SHARED physical pool.
    block_table  : 1-D int array, length ceil(seq_len / block_size).
        block_table[b] is the physical block backing this request's logical
        block b.
    seq_len      : number of VALID cached tokens for this request.
    block_size   : token slots per physical block.

    Gather the valid K/V vectors via block_table, then return
    softmax(K @ q / sqrt(d)) @ V as a (d,) vector.
    """
    raise NotImplementedError('your code here')
