import ref


def check(workdir):
    from specbench.measure import compute_acceptance_rate

    out = {"rates_matched": 0.0}
    ok = 0
    for i, p in enumerate(ref.PAIRS):
        want = ref.compute_acceptance_rate(p["draft"], p["target"], p["probs"])
        got = compute_acceptance_rate(p["draft"], p["target"], p["probs"])
        if abs(want - got) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"pair {i}: got {got}, reference {want}"
    out["rates_matched"] = float(ok)
    return out
