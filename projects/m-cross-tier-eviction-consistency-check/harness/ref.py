"""Reference implementation and deterministic fixture generator for harness."""

import numpy as np

def generate_scenarios(seed=42):
    rng = np.random.default_rng(seed)
    scenarios = []

    # Scenario 0: Valid aligned states
    scenarios.append(({
        "b1": {"hash": "h1", "dirty": False}
    }, {
        "b1": {"hash": "h1", "stale": False, "status": "VALID"}
    }, True, 0))

    # Scenario 1: Hash mismatch
    scenarios.append(({
        "b1": {"hash": "h1", "dirty": False}
    }, {
        "b1": {"hash": "h2", "stale": False, "status": "VALID"}
    }, False, 1))

    # Scenario 2: Dirty T0 with clean non-stale T1
    scenarios.append(({
        "b2": {"hash": "h2", "dirty": True}
    }, {
        "b2": {"hash": "h2", "stale": False, "status": "VALID"}
    }, False, 1))

    # Scenario 3: Unsynced eviction in T1
    scenarios.append(({}, {
        "b3": {"hash": "h3", "stale": True, "status": "EVICTED_DIRTY"}
    }, False, 1))

    # Scenario 4: Multiple violations
    scenarios.append(({
        "b1": {"hash": "h1", "dirty": True},
        "b2": {"hash": "h2", "dirty": False}
    }, {
        "b1": {"hash": "h1", "stale": False, "status": "VALID"},
        "b2": {"hash": "hx", "stale": False, "status": "VALID"},
        "b3": {"hash": "h3", "stale": True, "status": "EVICTED_DIRTY"}
    }, False, 3))

    return scenarios

def reference_checker(t0_state, t1_state):
    violations = []
    t0_ids = set(t0_state.keys())
    t1_ids = set(t1_state.keys())

    for block_id in t0_ids:
        t0_meta = t0_state[block_id]
        if block_id in t1_ids:
            t1_meta = t1_state[block_id]
            if t0_meta.get("hash") != t1_meta.get("hash"):
                violations.append((block_id, "HASH_MISMATCH"))
            if t0_meta.get("dirty", False) and not t1_meta.get("stale", False):
                violations.append((block_id, "DIRTY_T0_CLEAN_T1"))

    for block_id in t1_ids:
        t1_meta = t1_state[block_id]
        if t1_meta.get("status") == "EVICTED_DIRTY":
            violations.append((block_id, "UNSYNCED_EVICTION"))

    return len(violations) == 0, violations
