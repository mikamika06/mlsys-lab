def is_block_mapping_legal(seqs: list[dict]) -> bool:
    """Check a PagedAttention-style block table configuration for illegal
    physical-block aliasing.

    seqs: a list of live sequences, each a dict
        {"physical_block_ids": list[int], "is_shared": list[bool]}
    of equal length per sequence: `physical_block_ids[i]` is the physical
    block a sequence's logical block `i` maps to, and `is_shared[i]` says
    whether that mapping is a read-only, copy-on-write share (True) or a
    private, writable mapping (False) for THIS sequence.

    A physical block referenced by only one live sequence is always
    legal. A physical block referenced by TWO OR MORE distinct sequences
    is legal ONLY if every one of those references marks it `is_shared =
    True` (a read-only shared prefix, e.g. from radix-cache / APC
    prefix sharing). If even one of the sequences maps that same
    physical block as `is_shared = False` (a private, writable mapping)
    while another live sequence also maps it, that is illegal aliasing:
    a write through the private mapping would silently corrupt the other
    sequence's cache.

    Returns True iff the configuration has no illegal aliasing.
    """
    owners: dict[int, list[bool]] = {}
    for seq in seqs:
        ids = seq["physical_block_ids"]
        shared = seq["is_shared"]
        seen_this_seq: set[int] = set()
        for pid, is_shared in zip(ids, shared):
            if pid in seen_this_seq:
                continue  # a sequence referencing its own block twice isn't aliasing
            seen_this_seq.add(pid)
            owners.setdefault(pid, []).append(bool(is_shared))

    for pid, flags in owners.items():
        if len(flags) > 1 and not all(flags):
            return False
    return True
