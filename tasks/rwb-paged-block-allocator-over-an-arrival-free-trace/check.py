import math

import numpy as np


def _simulate(arrive_t, depart_t, seq_len, n_blocks, cost_fn):
    """Free-list block allocator over a time-ordered arrival/departure
    trace. At each request's arrival, it needs cost_fn(seq_len) blocks;
    admitted only if that many are currently free. Blocks are returned to
    the free pool at the (admitted) request's departure. On simultaneous
    events, departures are processed before arrivals at the same
    timestamp (freed capacity is available to same-instant arrivals).

    Returns (peak_blocks_used, admitted_count).
    """
    n = len(arrive_t)
    events = []
    for i in range(n):
        events.append((int(arrive_t[i]), 0, i))   # arrive: kind 0
        events.append((int(depart_t[i]), -1, i))  # depart: kind -1 (sorts first on ties)
    events.sort(key=lambda e: (e[0], e[1]))

    free = n_blocks
    used = 0
    admitted = [False] * n
    cost = [0] * n
    admitted_count = 0
    peak = 0

    for _t, kind, i in events:
        if kind == -1:
            if admitted[i]:
                free += cost[i]
                used -= cost[i]
        else:
            c = cost_fn(int(seq_len[i]))
            if c <= free:
                free -= c
                used += c
                admitted[i] = True
                cost[i] = c
                admitted_count += 1
                peak = max(peak, used)

    return peak, admitted_count


def _oracle(arrive_t, depart_t, seq_len, n_blocks, block_size, max_len):
    paged_cost = lambda L: math.ceil(L / block_size)
    contig_cost = lambda L: math.ceil(max_len / block_size)

    peak, admitted_paged = _simulate(arrive_t, depart_t, seq_len, n_blocks, paged_cost)
    _, admitted_contig = _simulate(arrive_t, depart_t, seq_len, n_blocks, contig_cost)
    return peak, admitted_paged, admitted_contig


def _fixture_configs():
    # (n_blocks, block_size); max_len fixed to the fixture's own worst case (64)
    return [(15, 8, 64), (25, 8, 64), (20, 16, 64), (30, 16, 64)]


def _synthetic_cases():
    rng = np.random.default_rng(17)
    cases = []
    for _ in range(3):
        n = int(rng.integers(10, 30))
        arrive_t = np.sort(rng.integers(0, 80, size=n))
        durations = rng.integers(4, 30, size=n)
        depart_t = arrive_t + durations
        max_len = int(rng.integers(32, 80))
        seq_len = rng.integers(1, max_len + 1, size=n)
        n_blocks = int(rng.integers(8, 25))
        block_size = int(rng.choice([4, 8, 16]))
        cases.append((arrive_t, depart_t, seq_len, n_blocks, block_size, max_len))
    return cases


def grade(sol, fx) -> dict:
    arrive_t = np.asarray(fx["arrive_t"])
    depart_t = np.asarray(fx["depart_t"])
    seq_len = np.asarray(fx["seq_len"])

    cases = [
        (arrive_t, depart_t, seq_len, n_blocks, block_size, max_len)
        for n_blocks, block_size, max_len in _fixture_configs()
    ] + _synthetic_cases()

    total = 0
    correct = 0
    for arrive_t_c, depart_t_c, seq_len_c, n_blocks, block_size, max_len in cases:
        total += 1
        ref = _oracle(arrive_t_c, depart_t_c, seq_len_c, n_blocks, block_size, max_len)
        try:
            got = sol.paged_allocator_trace(
                list(arrive_t_c), list(depart_t_c), list(seq_len_c), n_blocks, block_size, max_len
            )
            got = tuple(int(v) for v in got)
        except Exception:
            continue
        if got == ref:
            correct += 1

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
