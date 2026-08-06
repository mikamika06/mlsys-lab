import ref

def check(workdir):
    from preemption.overhead import compute_wasted_compute
    out = {"overhead_matched": 0.0}
    ok = 0
    total = len(ref.LOGS) * len(ref.SYSTEMS)
    for log in ref.LOGS:
        for s in ref.SYSTEMS:
            want = ref.compute_wasted_compute(log, ref.MODELS[0], s)
            got = compute_wasted_compute(log, ref.MODELS[0], s)
            if abs(got - want) < 1e-5:
                ok += 1
            else:
                out["_note"] = f"mismatch for log: got {got}, want {want}"
                return out
    if ok == total:
        out["overhead_matched"] = 1.0
    return out
