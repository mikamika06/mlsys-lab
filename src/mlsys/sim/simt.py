"""Deterministic SIMT (GPU-warp) memory model — the MLSYS VIRTUAL GPU.
Pure Python, identical on every machine. Models the memory-access behaviour that
GPU tasks actually reason about: global-memory coalescing, shared-memory bank
conflicts, and warp divergence — as exact, deterministic counts. A pinned spec.

Spec: warp = 32 lanes; shared memory = 32 banks of 4-byte words;
global memory served in fixed byte segments (default 128 B).
"""
WARP = 32
BANKS = 32
WORD_BYTES = 4


def coalesced_transactions(byte_addrs, segment=128):
    """Given the byte addresses one per active lane, return the number of
    distinct aligned memory segments touched = # of global transactions.
    1 == fully coalesced; up to len(addrs) == fully scattered."""
    return len({int(a) // segment for a in byte_addrs})


def bank_conflict_degree(word_indices, banks=BANKS):
    """Given the shared-memory WORD index per lane, return the n-way conflict
    degree = max over banks of the number of DISTINCT words requested in that
    bank. 1 == conflict-free (or broadcast of the same word)."""
    per_bank = {}
    for w in word_indices:
        per_bank.setdefault(int(w) % banks, set()).add(int(w))
    return max((len(s) for s in per_bank.values()), default=1)


def divergence_passes(predicates):
    """Given a bool predicate per lane, return how many passes the warp needs.
    1 == uniform (no divergence); 2 == both branch sides taken (serialised)."""
    return len({bool(p) for p in predicates}) or 1


def active_mask_passes(lane_active):
    """Number of distinct execution groups given a per-lane active flag list —
    a uniform generalisation used by masked / predicated kernels."""
    return len({bool(x) for x in lane_active}) or 1


def summarize_access(byte_addrs=None, word_indices=None, predicates=None, segment=128):
    """One-call summary for a warp's memory step."""
    out = {}
    if byte_addrs is not None:
        out["global_transactions"] = coalesced_transactions(byte_addrs, segment)
    if word_indices is not None:
        out["bank_conflict_degree"] = bank_conflict_degree(word_indices)
    if predicates is not None:
        out["divergence_passes"] = divergence_passes(predicates)
    return out
