import ref

def check(workdir):
    from ring.imbalance import analyze_naive_ring

    out = {"imbalance_correct": 0.0, "imbalance_total": 4.0}
    ok = 0
    for C in [2, 4, 8, 16]:
        want = ref.analyze_naive_ring(C)
        try:
            got = analyze_naive_ring(C)
            if got == want:
                ok += 1
            else:
                if "_note" not in out:
                    out["_note"] = f"failed for C={C}, got {got[:2]}, want {want[:2]}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"error for C={C}: {e}"
    out["imbalance_correct"] = float(ok)
    return out
