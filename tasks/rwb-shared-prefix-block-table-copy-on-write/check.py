import numpy as np


def _lcp_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _ceil_div(x, y):
    return -(-x // y)


def _oracle(tokens_a, tokens_b, block_size):
    P = _lcp_len(tokens_a, tokens_b)
    n_shared = _ceil_div(P, block_size)
    n_blocks_a = _ceil_div(len(tokens_a), block_size)
    n_blocks_b = _ceil_div(len(tokens_b), block_size)
    tail_a = n_blocks_a - n_shared
    tail_b = n_blocks_b - n_shared

    shared_ids = list(range(n_shared))
    nxt = n_shared
    tail_a_ids = list(range(nxt, nxt + tail_a))
    nxt += tail_a
    tail_b_ids = list(range(nxt, nxt + tail_b))
    nxt += tail_b

    block_table_a = shared_ids + tail_a_ids
    block_table_b = shared_ids + tail_b_ids
    return block_table_a, block_table_b, nxt


def _hand_cases():
    cases = []
    # P is an exact multiple of block_size.
    cases.append(([1, 2, 3, 4, 9, 9], [1, 2, 3, 4, 7, 7, 7], 2))
    # P is NOT a multiple of block_size (partial boundary block).
    cases.append(([1, 2, 3, 4, 5, 9, 9, 9], [1, 2, 3, 4, 5, 7, 7], 4))
    # Fully identical sequences.
    cases.append(([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 3))
    # No shared prefix at all.
    cases.append(([1, 2, 3], [9, 8, 7], 2))
    # One sequence is a strict prefix of the other.
    cases.append(([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7], 3))
    # Both empty.
    cases.append(([], [], 4))
    # One empty.
    cases.append(([], [1, 2, 3], 4))
    # block_size = 1.
    cases.append(([5, 5, 5, 6, 6], [5, 5, 5, 7], 1))
    return cases


def _gen_case(rng, vocab=50):
    block_size = int(rng.integers(1, 8))
    p_len = int(rng.integers(0, 20))
    prefix = list(rng.integers(0, vocab, size=p_len))
    tail_a_len = int(rng.integers(0, 15))
    tail_b_len = int(rng.integers(0, 15))
    tail_a = list(rng.integers(0, vocab, size=tail_a_len))
    tail_b = list(rng.integers(0, vocab, size=tail_b_len))
    # ensure the two tails don't accidentally start with the same token
    # (which would just extend the real shared prefix -- fine either way,
    # since the oracle recomputes P from the actual arrays regardless).
    return prefix + tail_a, prefix + tail_b, block_size


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(10):
        cases.append(_gen_case(rng))

    exact = 1.0
    for tokens_a, tokens_b, block_size in cases:
        ref = _oracle(list(tokens_a), list(tokens_b), block_size)
        try:
            got = sol.build_shared_prefix_block_tables(
                list(tokens_a), list(tokens_b), block_size
            )
            got_a, got_b, got_n = got
            got_a = list(int(x) for x in got_a)
            got_b = list(int(x) for x in got_b)
            got_n = int(got_n)
        except Exception:
            exact = 0.0
            break
        if (got_a, got_b, got_n) != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
