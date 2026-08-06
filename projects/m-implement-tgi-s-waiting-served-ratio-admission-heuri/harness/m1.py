import ref

def check(workdir):
    from router.admission import admit

    out = {"cases_matched": 0.0}
    ok = 0

    for i, (queue, active, max_total, max_prefill, wsr) in enumerate(ref.M1_CASES):
        want = ref.admit(queue, active, max_total, max_prefill, wsr)
        try:
            got = admit(queue, active, max_total, max_prefill, wsr)
        except Exception as e:
            out["_note"] = f"case {i} crashed: {e}"
            return out

        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"

    out["cases_matched"] = float(ok)
    return out
