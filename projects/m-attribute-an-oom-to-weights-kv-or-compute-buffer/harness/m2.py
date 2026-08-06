import ref

def check(workdir):
    out = {"oom_matched": 0.0}
    try:
        from memplan.oom import attribute_oom
    except ImportError:
        out["_note"] = "could not import memplan.oom"
        return out
    
    ok_oom = 0
    for args in ref.FIXTURES_OOM:
        want = ref.attribute_oom(*args)
        try:
            got = attribute_oom(*args)
            if got == want:
                ok_oom += 1
            else:
                out.setdefault("_note", f"failed on {args}: got {got}, want {want}")
        except Exception:
            pass
    if ref.FIXTURES_OOM:
        out["oom_matched"] = float(ok_oom) / len(ref.FIXTURES_OOM)
    return out
