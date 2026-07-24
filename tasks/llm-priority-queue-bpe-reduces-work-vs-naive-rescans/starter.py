def bpe_encode(ids, ranks):
    """Encode a symbol sequence with a BPE merge table using a priority queue.

    ``ids``   : list[int]           initial symbol ids (bytes 0..255 + merged ids).
    ``ranks`` : {(a, b): rank}      merging pair (a, b) yields id ``256 + rank``;
                                    lower rank = higher priority.

    Return the fully merged sequence. Repeatedly apply the lowest-rank adjacent
    merge until no adjacent pair has a rank. Use a heap so the work does not scale
    with (length * number of merges) — a full rescan per merge exceeds the op
    budget.
    """
    raise NotImplementedError("your code here")
