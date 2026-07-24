def compute_reuse_savings(seqs, block_size):
    """Total radix (token-level) vs block-level prefix reuse over a trace.

    seqs: list of N token-id sequences, processed in order; sequence i
        may reuse from any of seqs[0..i-1] (not itself, not later ones).
    block_size: positive int.

    For each i: radix_i = max over j<i of the token-exact longest common
    prefix between seqs[i] and seqs[j] (0 if none); block_i = radix_i
    rounded down to the nearest multiple of block_size.

    Returns (radix_saved_tokens, block_saved_tokens): radix_i and
    block_i summed over every sequence i in the trace.
    """
    raise NotImplementedError('your code here')
