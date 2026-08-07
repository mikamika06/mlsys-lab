import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from qformat.parse import parse_quant_config

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.parse_quant_config(cfg)
        try:
            got = parse_quant_config(cfg)
            if got == want:
                ok += 1
            else:
                out.setdefault("_note", f"config {i}: got {got}, want {want}")
        except Exception as e:
            out.setdefault("_note", f"config {i} threw {type(e).__name__}: {e}")
    out["configs_matched"] = float(ok)
    return out
