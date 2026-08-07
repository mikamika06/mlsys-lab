import ref


def check(workdir):
    from tensorgrid.reconstruct import reconstruct_length

    out = {"reconstruct_matched": 0.0, "total": float(len(ref.TEST_CASES))}
    ok = 0
    for i, (gs, bs) in enumerate(ref.TEST_CASES):
        want = ref.reconstruct_length(gs, bs)
        try:
            got = reconstruct_length(gs, bs)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}"
            continue
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    out["reconstruct_matched"] = float(ok)
    return out
