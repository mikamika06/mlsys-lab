import ref

def check(workdir):
    from export_fixer import unbake
    out = {"unbaked_matched": 0.0, "total": float(len(ref.GRAPHS))}
    ok = 0
    for g in ref.GRAPHS:
        want = ref.unbake_concat(g)
        try:
            got = unbake.unbake_concat(g)
            if got == want:
                ok += 1
        except Exception:
            pass
    out["unbaked_matched"] = float(ok)
    return out
