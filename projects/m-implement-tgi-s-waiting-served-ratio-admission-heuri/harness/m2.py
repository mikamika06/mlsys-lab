import ref

def check(workdir):
    from router.utilization import compute_utilization

    out = {"util_matched": 0.0}
    ok = 0

    for i, (log, max_total) in enumerate(ref.M2_LOGS):
        want = ref.compute_utilization(log, max_total)
        try:
            got = compute_utilization(log, max_total)
        except Exception as e:
            out["_note"] = f"log {i} crashed: {e}"
            return out

        if abs(want - got) < 1e-6:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"log {i}: got {got:.4f}, want {want:.4f}"

    out["util_matched"] = float(ok)
    return out
