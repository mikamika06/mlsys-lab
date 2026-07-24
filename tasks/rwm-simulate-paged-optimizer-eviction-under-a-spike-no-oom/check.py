from collections import OrderedDict

import numpy as np


def _oracle_paged_lru(trace, budget_pages):
    resident = OrderedDict()
    fault_count = 0
    evicted = []
    for page in trace:
        if page in resident:
            resident.move_to_end(page)  # hit -> now most-recently-used
        else:
            fault_count += 1
            if len(resident) >= budget_pages:
                evict_page, _ = resident.popitem(last=False)  # evict LRU
                evicted.append(evict_page)
            resident[page] = True
    return {
        "fault_count": fault_count,
        "evicted_pages": evicted,
        "final_resident": list(resident.keys()),
    }


def _build_cases():
    cases = []
    # hand-built: steady working set, then a spike of brand-new pages
    # that must evict the working set, then the working set returns
    # and must fault back in.
    cases.append(([0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2], 3))

    rng = np.random.default_rng(1)
    trace2 = rng.integers(0, 8, size=40).tolist()
    cases.append((trace2, 5))

    rng2 = np.random.default_rng(2)
    base = rng2.integers(0, 4, size=10).tolist()
    spike = list(range(100, 120))
    tail = base[:6]
    cases.append((base + spike + tail, 4))

    # budget of 1: every distinct page access evicts the previous one
    cases.append(([7, 7, 7, 8, 8, 9, 7, 9, 8], 1))

    return cases


def grade(sol, fx) -> dict:
    fault_match = 1.0
    evict_match = 1.0

    for trace, budget in _build_cases():
        ref = _oracle_paged_lru(trace, budget)

        try:
            got = sol.simulate_paged_eviction(list(trace), budget)
        except Exception:
            return {"fault_count_match": 0.0, "evicted_pages_match": 0.0}

        try:
            got_fault = int(got["fault_count"])
            got_evicted = list(got["evicted_pages"])
        except Exception:
            return {"fault_count_match": 0.0, "evicted_pages_match": 0.0}

        if got_fault != ref["fault_count"]:
            fault_match = 0.0
        if got_evicted != ref["evicted_pages"]:
            evict_match = 0.0

    return {"fault_count_match": fault_match, "evicted_pages_match": evict_match}
