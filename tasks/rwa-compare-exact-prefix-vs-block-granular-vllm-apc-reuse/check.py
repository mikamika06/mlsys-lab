import numpy as np


def _lcp_numpy(q, c) -> int:
    """Longest common prefix length of two int-id sequences, via NumPy:
    pad to equal length with a sentinel that cannot appear in either
    (so it never falsely matches), compare elementwise, and take the
    index of the first mismatch with argmax on the boolean array."""
    n = min(len(q), len(c))
    if n == 0:
        return 0
    qa = np.asarray(q[:n], dtype=np.int64)
    ca = np.asarray(c[:n], dtype=np.int64)
    eq = qa == ca
    if eq.all():
        return n
    return int(np.argmax(~eq))


def _ref_reuse(cache, queries, block_size):
    out = []
    for q in queries:
        best = 0
        for c in cache:
            lcp = _lcp_numpy(q, c)
            if lcp > best:
                best = lcp
        block = (best // block_size) * block_size
        out.append((best, block))
    return out


def _scenarios():
    scenarios = []

    # 1: single cache entry, clean divergence partway through a block
    cache = [list(range(37)) + [999, 999]]
    queries = [list(range(37)) + [500, 600]]
    scenarios.append((cache, queries, 16))

    # 2: multiple cache entries, best match is NOT the first one
    cache = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 999],
        [1, 2, 9, 9, 9],
    ]
    queries = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 0, 0],
    ]
    scenarios.append((cache, queries, 4))

    # 3: query shorter than its best-matching cache entry (full match)
    cache = [list(range(100))]
    queries = [list(range(20))]
    scenarios.append((cache, queries, 8))

    # 4: no match at all (diverges at token 0), and empty cache
    cache = [[5, 5, 5, 5, 5, 5]]
    queries = [[1, 2, 3], []]
    scenarios.append((cache, queries, 4))
    scenarios.append(([], [[1, 2, 3], []], 4))

    # 5: exact multiple of block_size (block_reuse == exact_reuse)
    cache = [list(range(32))]
    queries = [list(range(32)) + [7, 7]]
    scenarios.append((cache, queries, 16))

    # 6: block_size == 1 (block reuse degenerates to exact reuse)
    cache = [[1, 2, 3, 4, 5, 9, 9]]
    queries = [[1, 2, 3, 4, 0, 0]]
    scenarios.append((cache, queries, 1))

    # 7: seeded random sequences with shared random prefixes
    rng = np.random.default_rng(0)
    vocab = 50
    base = rng.integers(0, vocab, size=64).tolist()
    cache = []
    for cut in (5, 20, 40, 64):
        entry = base[:cut] + rng.integers(0, vocab, size=10).tolist()
        cache.append(entry)
    queries = []
    for cut in (0, 3, 22, 41, 63, 64):
        qq = base[:cut] + rng.integers(0, vocab, size=6).tolist()
        queries.append(qq)
    scenarios.append((cache, queries, 8))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0
    for cache, queries, block_size in _scenarios():
        ref = _ref_reuse(cache, queries, block_size)
        try:
            got = sol.prefix_reuse_lengths(
                [list(c) for c in cache],
                [list(q) for q in queries],
                block_size,
            )
        except Exception:
            total += len(queries)
            continue

        try:
            if len(got) != len(ref):
                total += len(queries)
                continue
            for g, r in zip(got, ref):
                total += 1
                ge, gb = int(g[0]), int(g[1])
                if ge == r[0] and gb == r[1]:
                    correct += 1
        except Exception:
            total += len(queries)
            continue

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
