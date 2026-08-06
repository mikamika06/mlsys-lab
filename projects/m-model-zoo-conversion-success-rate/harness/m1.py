import ref


def check(workdir):
    from conv.reader import read_signature

    out = {"signatures_matched": 0.0, "total": float(len(ref.MODELS))}
    ok = 0
    for i, m in enumerate(ref.MODELS):
        want = ref.read_signature(m["signature_bytes"])
        got = read_signature(m["signature_bytes"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: got {got}, reference {want}"
    out["signatures_matched"] = float(ok)
    return out
