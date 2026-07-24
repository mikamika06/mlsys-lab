import numpy as np


def _block_stats(requests, block_size):
    """Real automatic-prefix-caching (APC) simulation: requests are
    processed in order against one shared cache. A block is a HIT
    (reused) only if every block before it in this request was also a
    hit, and the exact prefix of tokens through the end of this block
    already exists in the cache from an earlier request. The first
    non-matching block, and every block after it in this request, is a
    MISS (computed) -- prefix-cache reuse only ever extends an
    exact-matching prefix. A request's own blocks become visible to the
    cache only after it finishes."""
    cache = set()
    reused = 0
    computed = 0
    for tokens in requests:
        tokens = list(tokens)
        n = len(tokens)
        num_blocks = (n + block_size - 1) // block_size
        still_reusing = True
        boundary_prefixes = []
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, n)
            prefix = tuple(tokens[:end])
            boundary_prefixes.append(prefix)
            if still_reusing and prefix in cache:
                reused += 1
            else:
                still_reusing = False
                computed += 1
        for prefix in boundary_prefixes:
            cache.add(prefix)
    return reused, computed


def _hand_cases():
    cases = []

    rng = np.random.default_rng(1)
    vocab = 300
    req1 = list(rng.integers(0, vocab, size=32))
    req2 = req1[:16] + list(rng.integers(0, vocab, size=20))  # shares 2 blocks of 8
    cases.append(([req1, req2], 8))

    rng2 = np.random.default_rng(2)
    req = list(rng2.integers(0, vocab, size=24))
    cases.append(([req, list(req), list(req)], 8))

    rng3 = np.random.default_rng(3)
    reqs = [list(rng3.integers(0, vocab, size=int(rng3.integers(5, 30)))) for _ in range(5)]
    cases.append((reqs, 6))

    cases.append(([], 4))

    cases.append(([[1, 2, 3]], 4))

    return cases


def _gen_case(rng, vocab=500):
    block_size = int(rng.choice([2, 4, 8, 16]))
    requests = []
    base = list(rng.integers(0, vocab, size=int(rng.integers(10, 60))))
    requests.append(base)
    n_children = int(rng.integers(2, 6))
    for _ in range(n_children):
        prev = requests[-1]
        if len(prev) == 0 or rng.random() < 0.15:
            requests.append(list(rng.integers(0, vocab, size=int(rng.integers(5, 40)))))
            continue
        shared = int(rng.integers(0, len(prev) + 1))
        tail_len = int(rng.integers(0, 30))
        tail = list(rng.integers(0, vocab, size=tail_len))
        requests.append(prev[:shared] + tail)
    return requests, block_size


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(10):
        cases.append(_gen_case(rng))

    exact = 1.0
    for requests, block_size in cases:
        ref = _block_stats(requests, block_size)
        req_copy = [list(r) for r in requests]
        try:
            got = sol.prefix_cache_block_stats(req_copy, block_size)
            got = (int(got[0]), int(got[1]))
        except Exception:
            exact = 0.0
            break
        if got != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
