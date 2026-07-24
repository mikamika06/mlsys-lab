import numpy as np


def _reused_tokens(requests, block_size):
    """Real prefix-cache simulation: requests are processed in order
    against one shared cache. A request's block i is a hit only if every
    block before it in that request was also a hit (its tokens up to the
    end of block i exactly equal a prefix already produced by an earlier
    request) -- the first divergence stops all further reuse for that
    request. Hit blocks are added to the cache only after the whole
    request finishes, so a request never reuses its own blocks."""
    cache = set()
    total = 0
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
                total += end - start
            else:
                still_reusing = False
        for prefix in boundary_prefixes:
            cache.add(prefix)
    return total


def _oracle(requests, block_size_a, block_size_b):
    ra = _reused_tokens(requests, block_size_a)
    rb = _reused_tokens(requests, block_size_b)
    if ra > rb:
        better = "a"
    elif rb > ra:
        better = "b"
    else:
        better = "tie"
    return (ra, rb, better)


def _hand_cases():
    # Two requests sharing a 20-token prefix then diverging. block_size=4
    # divides 20 evenly (5 full blocks reused, 20 tokens); block_size=16
    # does not (the block covering tokens[16:32] straddles the divergence
    # at 20 and is wholly wasted, only the first block of 16 is reused).
    rng = np.random.default_rng(1)
    vocab = 300
    req1 = list(rng.integers(0, vocab, size=64))
    req2 = req1[:20] + list(rng.integers(0, vocab, size=30))
    case_a = ([req1, req2], 4, 16)

    # Identical requests: full reuse regardless of block size (both should
    # report the same total minus the un-cacheable first occurrence).
    rng2 = np.random.default_rng(2)
    req = list(rng2.integers(0, vocab, size=48))
    case_b = ([req, list(req), list(req)], 8, 24)

    # No shared prefixes at all: zero reuse for either block size.
    rng3 = np.random.default_rng(3)
    reqs = [list(rng3.integers(0, vocab, size=int(rng3.integers(10, 40)))) for _ in range(4)]
    case_c = (reqs, 8, 8)

    return [case_a, case_b, case_c]


def _gen_case(rng, vocab=500):
    block_size_a = int(rng.choice([2, 4, 8]))
    block_size_b = int(rng.choice([8, 16, 32]))
    if block_size_b <= block_size_a:
        block_size_b = block_size_a * 4

    requests = []
    base = list(rng.integers(0, vocab, size=int(rng.integers(20, 60))))
    requests.append(base)
    n_children = int(rng.integers(2, 5))
    for _ in range(n_children):
        prev = requests[-1]
        shared = int(rng.integers(1, len(prev)))
        tail_len = int(rng.integers(5, 40))
        tail = list(rng.integers(0, vocab, size=tail_len))
        requests.append(prev[:shared] + tail)
    return requests, block_size_a, block_size_b


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(8):
        cases.append(_gen_case(rng))

    exact = 1.0
    for requests, block_size_a, block_size_b in cases:
        ref = _oracle(requests, block_size_a, block_size_b)
        req_copy = [list(r) for r in requests]
        try:
            got = sol.block_size_reuse_comparison(req_copy, block_size_a, block_size_b)
        except Exception:
            exact = 0.0
            break
        try:
            got = tuple(got)
        except Exception:
            exact = 0.0
            break
        if len(got) != 3:
            exact = 0.0
            break
        if (int(got[0]), int(got[1]), str(got[2])) != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
