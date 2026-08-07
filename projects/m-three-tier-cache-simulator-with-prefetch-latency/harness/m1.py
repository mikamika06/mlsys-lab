import ref

def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    try:
        from cachesim.simulator import simulate
    except ImportError:
        return {"_note": "Failed to import cachesim.simulator"}

    out = {"latency_rel_err": 0.0, "evictions_match": 1.0}
    max_err = 0.0

    for i, trace in enumerate(ref.TRACES[:2]):
        want = ref.simulate(trace, 1024, 2048, "always", "wb")
        try:
            got = simulate(trace, 1024, 2048, "always", "wb")
        except NotImplementedError:
            return {"_note": "simulate not implemented"}

        err = abs(got["latency_ns"] - want["latency_ns"]) / max(want["latency_ns"], 1)
        max_err = max(max_err, err)

        if got["l1_evictions"] != want["l1_evictions"] or got["l2_evictions"] != want["l2_evictions"]:
            out["evictions_match"] = 0.0
            out["_note"] = f"Trace {i}: Expected evicts {want['l1_evictions']}/{want['l2_evictions']}, got {got['l1_evictions']}/{got['l2_evictions']}"

    out["latency_rel_err"] = max_err
    return out
