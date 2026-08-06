import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from accounting.sizes import optimize_model
    except ImportError:
        return {"matched_plan": 0.0, "matched_total": 0.0}

    out = {"matched_plan": 0.0, "matched_total": 0.0}

    w_p, w_t = ref.optimize_model(ref.SHAPES, ref.METHODS)
    try:
        g_p, g_t = optimize_model(ref.SHAPES, ref.METHODS)
        if g_p == w_p:
            out["matched_plan"] += 1.0
        else:
            out.setdefault("_note", f"plan 1 mismatch: got {g_p}, want {w_p}")
        if g_t == w_t:
            out["matched_total"] += 1.0
        else:
            out.setdefault("_note", f"total 1 mismatch: got {g_t}, want {w_t}")
    except Exception as e:
        out.setdefault("_note", f"test 1 raised {type(e).__name__}")

    allowed2 = ["float16", "lut4_channel_fp16"]
    w_p, w_t = ref.optimize_model(ref.SHAPES, allowed2)
    try:
        g_p, g_t = optimize_model(ref.SHAPES, allowed2)
        if g_p == w_p:
            out["matched_plan"] += 1.0
        if g_t == w_t:
            out["matched_total"] += 1.0
    except Exception:
        pass

    return out
