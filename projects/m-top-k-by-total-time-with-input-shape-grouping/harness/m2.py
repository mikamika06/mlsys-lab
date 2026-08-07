import os
import ref


def check(workdir):
    import analyzer.profiler as profiler
    out = {"match": 0.0, "total": 5.0}
    ok = 0

    for i in range(5):
        path = os.path.join(workdir, f"trace_m2_{i}.json")
        ref.generate_trace(path, seed=100+i, num_events=1000)

        with open(path, "r") as f:
            import json
            events = json.load(f)["traceEvents"]

        want_agg = ref.aggregate_by_shape(events)
        want = ref.top_k_by_total_time(want_agg, k=3)

        try:
            got = profiler.top_k_by_total_time(events, k=3)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Crash on trace {i}: {e}"
            continue

        if got == want:
            ok += 1
        else:
            if "_note" not in out:
                out["_note"] = f"Mismatch on trace {i}"

    out["match"] = float(ok)
    return out
