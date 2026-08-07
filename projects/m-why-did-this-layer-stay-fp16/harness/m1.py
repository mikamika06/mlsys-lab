import ref

def check(workdir):
    from quantopt.analysis import why_fp16

    layers = ref.make_layers()
    ok = 0
    out = {"reasons_matched": 0.0}
    for l in layers:
        want = ref.analyze_fp16(l)
        got = why_fp16(l)
        if got == want:
            ok += 1
    out["reasons_matched"] = float(ok)
    return out
