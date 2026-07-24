import numpy as np


def _ref_simulate(trace, capacity, followup):
    nodes = {}        # prefix tuple -> last_touch clock value
    child_count = {}  # prefix tuple -> number of live children
    clock = 0
    evicted_total = 0

    def touch_path(seq):
        nonlocal clock
        prefix = ()
        for tok in seq:
            prefix = prefix + (tok,)
            clock += 1
            if prefix not in nodes:
                parent = prefix[:-1]
                child_count[parent] = child_count.get(parent, 0) + 1
                child_count.setdefault(prefix, 0)
            nodes[prefix] = clock

    def evict_to_capacity():
        nonlocal evicted_total
        while len(nodes) > capacity:
            leaf = None
            best_t = None
            for p, t in nodes.items():
                if child_count.get(p, 0) == 0:
                    if best_t is None or t < best_t:
                        best_t = t
                        leaf = p
            del nodes[leaf]
            child_count.pop(leaf, None)
            parent = leaf[:-1]
            child_count[parent] -= 1
            evicted_total += 1

    for _op, seq in trace:
        touch_path(tuple(seq))
        evict_to_capacity()

    hits = sum(1 for seq in followup if tuple(seq) in nodes)
    hit_rate = hits / len(followup) if followup else 0.0

    return evicted_total, hit_rate


def _gen_random_scenario(seed, n_inserts, max_len, vocab, capacity):
    rng = np.random.default_rng(seed)
    trace = []
    inserted = []
    for _ in range(n_inserts):
        L = int(rng.integers(1, max_len + 1))
        seq = rng.integers(0, vocab, size=L).tolist()
        trace.append(("insert", seq))
        inserted.append(seq)
        if rng.random() < 0.3 and inserted:
            # re-query a previously inserted sequence (refreshes recency)
            q = inserted[int(rng.integers(0, len(inserted)))]
            trace.append(("query", q))
    followup = inserted[:3] + inserted[-3:]
    return trace, capacity, followup


def _scenarios():
    scenarios = []

    # 1: no eviction ever (capacity comfortably large)
    trace1 = [
        ("insert", [1, 2, 3]),
        ("insert", [1, 2, 4]),
        ("insert", [1, 5]),
        ("query", [1, 2, 3]),
    ]
    scenarios.append((trace1, 10, [[1, 2, 3], [1, 2, 4], [1, 5], [9, 9]]))

    # 2: from the task.md example -- single eviction
    trace2 = [
        ("insert", [1, 2, 3]),
        ("insert", [1, 2, 4]),
        ("insert", [1, 5]),
    ]
    scenarios.append((trace2, 4, [[1, 2, 3], [1, 2, 4], [1, 5]]))

    # 3: tight capacity forces multiple evictions within one insert
    trace3 = [
        ("insert", [1, 2, 3, 4, 5, 6]),
    ]
    scenarios.append((trace3, 2, [[1], [1, 2], [1, 2, 3, 4, 5, 6]]))

    # 4: query refreshes recency and changes eviction choice
    trace4 = [
        ("insert", [1, 2]),   # nodes (1,) (1,2)
        ("insert", [3, 4]),   # nodes (3,) (3,4)
        ("query", [1, 2]),    # refresh (1,) (1,2) -- now (3,4) branch is older
        ("insert", [5, 6]),   # forces eviction: capacity=4, now have 6 nodes -> evict 2
    ]
    scenarios.append((trace4, 4, [[1, 2], [3, 4], [5, 6]]))

    # 5: capacity=1, every insert immediately collapses to one node
    trace5 = [
        ("insert", [1, 2, 3]),
        ("insert", [4, 5]),
        ("insert", [6]),
    ]
    scenarios.append((trace5, 1, [[6], [1, 2, 3], [4, 5]]))

    # 6+: seeded random scenarios over a small shared vocabulary
    scenarios.append(_gen_random_scenario(0, n_inserts=12, max_len=4, vocab=5, capacity=10))
    scenarios.append(_gen_random_scenario(1, n_inserts=20, max_len=3, vocab=4, capacity=6))
    scenarios.append(_gen_random_scenario(2, n_inserts=15, max_len=5, vocab=6, capacity=15))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0

    for trace, capacity, followup in _scenarios():
        total += 1
        ref_evicted, ref_hit = _ref_simulate(trace, capacity, followup)

        trace_arg = [(op, list(seq)) for op, seq in trace]
        try:
            got_evicted, got_hit = sol.radix_lru_cache(trace_arg, capacity, [list(s) for s in followup])
        except Exception:
            continue

        try:
            got_evicted = int(got_evicted)
            got_hit = float(got_hit)
        except Exception:
            continue

        if got_evicted != ref_evicted:
            continue
        if abs(got_hit - ref_hit) > 1e-9:
            continue

        correct += 1

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
