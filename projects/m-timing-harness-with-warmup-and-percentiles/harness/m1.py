import ref

def check(workdir):
    import time
    import sys

    clock = ref.SimClock()
    old_time = time.perf_counter_ns
    time.perf_counter_ns = clock

    try:
        sys.path.insert(0, workdir)
        sys.modules.pop("bench.harness", None)
        from bench.harness import benchmark

        out = {"works": 0.0, "rel_err": 1.0}

        op1 = ref.MockOp(clock, [100]*10 + [200]*10 + [300]*10)
        clock.now = 0
        got = benchmark(op1, 5, 20, [50, 90, 99])

        op1_ref = ref.MockOp(clock, [100]*10 + [200]*10 + [300]*10)
        clock.now = 0
        want = ref.benchmark(op1_ref, 5, 20, [50, 90, 99])

        if got.keys() != want.keys():
            out["_note"] = f"keys mismatch: {got.keys()} != {want.keys()}"
            return out

        err = 0.0
        for k in want:
            err += abs(got[k] - want[k]) / (want[k] + 1e-9)

        out["rel_err"] = err
        if err < 1e-3:
            out["works"] = 1.0

    finally:
        time.perf_counter_ns = old_time
        sys.path.pop(0)

    return out
