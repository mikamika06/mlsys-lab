def prefix_tokens_saved(seqs: list[list[int]]) -> tuple[int, int]:
    """
    Insert `seqs` (each a list of token ids) one at a time into a shared
    radix/prefix tree, in order -- exactly the structure RadixAttention /
    prefix caching uses to reuse KV cache across requests.

    For each sequence, walk from the tree's root as far as an exact
    matching path already exists (built by previously inserted
    sequences): those leading tokens are "saved" (their KV is already
    cached, no need to recompute). The remaining, non-matching tail is
    "computed" (fresh prefill). After counting, graft that remaining
    tail onto the tree as new nodes so later sequences can reuse it too.

    Returns (total_saved, total_computed), summed over every sequence.
    """
    root: dict = {}
    total_saved = 0
    total_computed = 0

    for seq in seqs:
        node = root
        i = 0
        while i < len(seq) and seq[i] in node:
            node = node[seq[i]]
            i += 1

        total_saved += i
        total_computed += len(seq) - i

        for j in range(i, len(seq)):
            child: dict = {}
            node[seq[j]] = child
            node = child

    return total_saved, total_computed
