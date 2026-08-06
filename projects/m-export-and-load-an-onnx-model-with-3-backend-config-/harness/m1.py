import ref

def check(workdir):
    from onnxconfig.exporter import build_base_config
    out = {"structure_matched": 0.0}
    ok = 0
    for spec in ref.MODELS:
        want = ref.get_base_config(spec)
        got = build_base_config(spec)
        if got == want:
            ok += 1
    if ok == len(ref.MODELS):
        out["structure_matched"] = 1.0
    return out
