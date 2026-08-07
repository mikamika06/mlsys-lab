import ref


def check(workdir):
    from trtcache.parser import is_reusable

    out = {"decisions_matched": 0.0, "total": float(len(ref.CONFIGS))}
    ok = 0
    for i, (ch, bc) in enumerate(ref.CONFIGS):
        want = ref.ref_reusable(ch, bc)
        got = is_reusable(ch, bc)
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, want {want}"

    out["decisions_matched"] = float(ok)
    return out
