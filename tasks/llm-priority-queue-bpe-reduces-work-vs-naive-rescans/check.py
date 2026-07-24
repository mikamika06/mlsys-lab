import numpy as np

from mlsys import probe


def _oracle_encode(ids, ranks):
    """Straightforward round-based BPE encode (the reference definition).

    Each round: find the adjacent pair with the smallest rank present in the
    sequence, then replace every non-overlapping occurrence of it (left to right)
    with ``256 + rank``. Repeat until no adjacent pair has a rank.
    """
    word = list(ids)
    while len(word) >= 2:
        best = None
        best_rank = None
        for i in range(len(word) - 1):
            r = ranks.get((word[i], word[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best = (word[i], word[i + 1])
        if best is None:
            break
        first, second = best
        new_id = 256 + best_rank
        merged = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                merged.append(new_id)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        word = merged
    return word


def _train(seq, num_merges):
    """Greedy BPE training -> a VALID merge table (respects the rank invariant).

    Repeatedly merge the most frequent adjacent pair, assigning it the next rank
    and the id ``256 + rank``. Because merged ids only ever appear after the step
    that created them, no consuming merge can outrank a producing merge.
    """
    ids = list(seq)
    ranks = {}
    r = 0
    for _ in range(num_merges):
        counts = {}
        for i in range(len(ids) - 1):
            p = (ids[i], ids[i + 1])
            counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        # most frequent; deterministic tie-break on the pair
        best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        ranks[best] = r
        new_id = 256 + r
        first, second = best
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == first and ids[i + 1] == second:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        ids = merged
        r += 1
    return ranks


def _make_cases():
    rng = np.random.default_rng(0)
    cases = []
    # Large cases: exercise the op-count budget (naive rescans blow past it).
    for length, alphabet, m in [(400, 6, 60), (350, 5, 45), (500, 8, 80), (300, 6, 50)]:
        seq = rng.integers(0, alphabet, size=length).tolist()
        cases.append((seq, _train(seq, m)))
    # Edge cases: correctness only.
    cases.append(([], {}))                       # empty
    cases.append(([65], {(65, 66): 0}))          # single symbol, nothing to merge
    cases.append(([65, 66], {}))                 # no applicable merge
    cases.append(([65, 66], {(65, 66): 0}))      # one merge
    cases.append(([7, 7, 7], {(7, 7): 0}))       # overlapping occurrences
    return cases


def grade(sol, fx) -> dict:
    cases = _make_cases()
    exact = 1.0
    op_count = 0.0
    for seq, ranks in cases:
        ref = _oracle_encode(seq, ranks)
        try:
            got = list(sol.bpe_encode(list(seq), dict(ranks)))
            events = probe.count_line_events(sol.bpe_encode, list(seq), dict(ranks))
        except Exception:
            return {"exact_match": 0.0, "op_count": float("inf")}
        if got != ref:
            exact = 0.0
        if events > op_count:
            op_count = float(events)
    return {"exact_match": exact, "op_count": op_count}
