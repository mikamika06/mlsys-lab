import numpy as np


def _lcp(q, c) -> int:
    n = min(len(q), len(c))
    i = 0
    while i < n and q[i] == c[i]:
        i += 1
    return i


def _ref_savings(seqs, block_size):
    radix_total = 0
    block_total = 0
    for i in range(len(seqs)):
        best = 0
        for j in range(i):
            lcp = _lcp(seqs[i], seqs[j])
            if lcp > best:
                best = lcp
        radix_total += best
        block_total += (best // block_size) * block_size
    return radix_total, block_total


def _scenarios():
    scenarios = []

    # diverges partway through a block
    scenarios.append(([
        list(range(50)),
        list(range(37)) + [999, 999],
        list(range(20)) + [500],
    ], 16))

    # best match is not the immediately preceding sequence
    scenarios.append(([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 9, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
    ], 4))

    # no shared prefixes at all
    scenarios.append(([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], 8))

    # exact multiples of block_size, and a later seq matching an earlier one exactly
    scenarios.append(([
        list(range(32)),
        list(range(32)) + [1, 1],
        list(range(16)) + [2, 2],
    ], 16))

    # block_size = 1 (block reuse degenerates to radix reuse)
    scenarios.append(([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 9, 9],
        [1, 2, 9, 9, 9],
    ], 1))

    # single sequence (no prior cache at all)
    scenarios.append(([[1, 2, 3, 4]], 4))

    rng = np.random.default_rng(0)
    vocab = 40
    base = rng.integers(0, vocab, size=80).tolist()
    seqs = []
    for cut in (0, 5, 22, 41, 63, 80, 30, 71):
        seq = base[:cut] + rng.integers(0, vocab, size=8).tolist()
        seqs.append(seq)
    scenarios.append((seqs, 8))
    scenarios.append((seqs, 32))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0

    for seqs, block_size in _scenarios():
        total += 1
        radix_ref, block_ref = _ref_savings(seqs, block_size)
        if radix_ref < block_ref:
            continue  # broken scenario, shouldn't happen

        try:
            radix_got, block_got = sol.compute_reuse_savings(
                [list(s) for s in seqs], block_size,
            )
        except Exception:
            continue

        try:
            radix_got = int(radix_got)
            block_got = int(block_got)
        except Exception:
            continue

        if radix_got < block_got:
            continue
        if radix_got == radix_ref and block_got == block_ref:
            correct += 1

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
