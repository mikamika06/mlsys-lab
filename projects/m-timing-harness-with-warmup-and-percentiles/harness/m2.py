import ref

def check(workdir):
    import time
    import sys
    import random

    clock = ref.SimClock()
    old_time = time.perf_counter_ns
    time.perf_counter_ns = clock

    try:
        sys.path.insert(0, workdir)
        sys.modules.pop("bench.harness", None)
        from bench.harness import find_stable_iters

        out = {"iters_match": 0.0}

        rng = random.Random(42)
        seq = [rng.randint(1000, 2000) for _ in range(10000)]

        op = ref.MockOp(clock, seq)
        clock.now = 0
        got = find_stable_iters(op, 0.05)

        op_ref = ref.MockOp(clock, seq)
        clock.now = 0
        want = ref.find_stable_iters(op_ref, 0.05)

        if got == want:
            out["iters_match"] = 1.0
        else:
            out["_note"] = f"got {got} iters, want {want}"

    finally:
        time.perf_counter_ns = old_time
        sys.path.pop(0)

    return out
