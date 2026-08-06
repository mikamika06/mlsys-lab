import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from accounting.sizes import layer_bytes
    except ImportError:
        return {"matched": 0.0, "_note": "could not import layer_bytes"}

    out = {"matched": 0.0}
    ok = 0
    for shape in ref.SHAPES:
        for m in ref.METHODS:
            want = ref.layer_bytes(shape, m)
            try:
                got = layer_bytes(shape, m)
                if want == got:
                    ok += 1
                else:
                    out.setdefault("_note", f"shape {shape}, method {m}: got {got}, want {want}")
            except Exception as e:
                out.setdefault("_note", f"shape {shape}, method {m}: raised {type(e).__name__}")

    out["matched"] = float(ok)
    return out
