import ref


def check(workdir):
    from cacheopt.reorder import reorder_requests

    out = {"reorder_matched": 0.0}
    ok = 0
    for i, reqs in enumerate(ref.CONFIGS):
        want = ref.reorder_requests(reqs, 10)
        got = reorder_requests(reqs, 10)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, want {want}"
    out["reorder_matched"] = float(ok)
    return out
