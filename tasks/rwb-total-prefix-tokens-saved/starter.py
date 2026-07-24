def prefix_tokens_saved(seqs: list[list[int]]) -> tuple[int, int]:
    """
    Insert `seqs` (each a list of token ids) one at a time into a shared
    radix/prefix tree, in order.

    For each sequence, count how many leading tokens match an existing
    tree path ("saved") vs how many are new ("computed"), then graft the
    new tail onto the tree so later sequences can reuse it too.

    Returns (total_saved, total_computed), summed over every sequence.
    """
    raise NotImplementedError('your code here')
