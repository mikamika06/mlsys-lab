def cow_kv_branches(prompt_kv, branch_kvs, block_size):
    """
    Simulate a block-paged KV cache with copy-on-write (COW) prefix sharing.

    prompt_kv: (P, D) float array -- KV vectors for the P shared prompt tokens.
    branch_kvs: list of B arrays, branch_kvs[b] has shape (L_b, D) -- the KV
      vectors branch b appends after the shared prompt, in order.
    block_size: number of KV-vector slots per physical block.

    All B branches start by referencing the SAME physical blocks that hold
    prompt_kv: P // block_size full blocks plus (if P % block_size != 0) one
    shared PARTIAL last block. A full block is immutable once full -- a
    branch appending past it always starts a fresh block. A partial block
    may be written into directly only if the writer currently holds the
    sole reference to it (reference count == 1); otherwise the writer must
    copy it to a brand-new physical block first (copy-on-write), drop its
    reference to the old block, and write into the copy. Process branches
    in order 0..B-1, and within a branch process its tokens in order.

    Return (branch_sequences, total_blocks_allocated):
      - branch_sequences[b] is the (P + L_b, D) reconstructed KV sequence
        for branch b (prompt followed by its own appended tokens).
      - total_blocks_allocated is the total count of distinct physical
        blocks ever allocated across the whole simulation (the initial
        prompt blocks plus every COW copy and every fresh block).
    """
    raise NotImplementedError('your code here')
