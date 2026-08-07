import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from zero3.schedule import build_schedule, simulate_peak_memory
    except ImportError:
        return {"_note": "could not import zero3.schedule functions"}

    out = {"schedule_matched": 0.0, "simulate_matched": 0.0}

    ok_sched = 0
    test_cases = [(4, 1), (10, 2), (5, 0)]
    for n, p in test_cases:
        want = ref.build_schedule(n, p)
        try:
            got = build_schedule(n, p)
            if got == want:
                ok_sched += 1
            else:
                out["_note"] = f"schedule mismatch for n={n}, prefetch={p}"
                break
        except Exception as e:
            out["_note"] = f"build_schedule exception: {e}"
            break

    if ok_sched == len(test_cases):
        out["schedule_matched"] = 1.0

    ok_sim = 0
    for layers, _ in ref.CONFIGS:
        sched = ref.build_schedule(len(layers), 1)
        want = ref.simulate_peak_memory(layers, sched)
        try:
            got = simulate_peak_memory(layers, sched)
            if got == want:
                ok_sim += 1
            else:
                out["_note"] = out.get("_note", "") + f" simulate mismatch: got {got}, want {want}"
                break
        except Exception as e:
            out["_note"] = out.get("_note", "") + f" simulate exception: {e}"
            break

    if ok_sim == len(ref.CONFIGS):
        out["simulate_matched"] = 1.0

    sys.path.pop(0)
    return out
