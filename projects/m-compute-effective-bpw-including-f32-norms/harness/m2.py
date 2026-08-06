import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from effbpw.budget import select_quantization
    except ImportError:
        return {"rel_err": 1.0, "selected_match": 0.0, "_note": "failed to import select_quantization"}

    out = {"rel_err": 0.0, "selected_match": 1.0}
    max_err = 0.0
    ok = True

    for fixture in ref.FIXTURES:
        t_shapes, quants, ctx, layers, kv_heads, h_dim, vram = fixture
        want = ref.select_quantization(t_shapes, quants, ctx, layers, kv_heads, h_dim, vram)
        try:
            got = select_quantization(t_shapes, quants, ctx, layers, kv_heads, h_dim, vram)
        except NotImplementedError:
            return {"rel_err": 1.0, "selected_match": 0.0, "_note": "NotImplementedError raised"}

        if want["selected_quant"] != got.get("selected_quant"):
            ok = False
            out["_note"] = f"Expected {want['selected_quant']}, got {got.get('selected_quant')}"

        got_ratios = got.get("size_ratios", {})
        for q_name, w_ratio in want["size_ratios"].items():
            g_ratio = got_ratios.get(q_name, 0.0)
            err = abs(w_ratio - g_ratio) / max(1e-9, w_ratio)
            max_err = max(max_err, err)

    out["rel_err"] = float(max_err)
    out["selected_match"] = 1.0 if ok else 0.0
    return out
