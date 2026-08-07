import ref

def check(workdir):
    from trtplug.analysis import analyze_path

    out = {"analysis_matched": 0.0}
    ok = 0
    for model in ref.MODELS:
        want = ref.analyze_path(model) if hasattr(ref, "analyze_path") else {}
        # compute inline expected if ref doesn't have wrapper
        nodes = model.get("nodes", [])
        dc = sum(n.get("flops", 0) * 1.5 for n in nodes)
        pc = sum(n.get("flops", 0) * 0.8 for n in nodes) + 1000
        expected = {
            "decomposed_cost": float(dc),
            "plugin_cost": float(pc),
            "overhead_diff": float(abs(dc - pc)),
            "recommendation": "plugin" if pc < dc else "decomposed"
        }
        try:
            got = analyze_path(model)
        except Exception:
            got = {}
        if got == expected:
            ok += 1
    out["analysis_matched"] = float(ok)
    return out
