import ref

def check(workdir):
    from ort_perf.env import check_shadow_wheel

    out = {"cases_matched": 0.0}
    cases = [
        {"onnxruntime": "1.14.0"},
        {"onnxruntime-gpu": "1.14.0", "numpy": "1.21.0"},
        {"onnxruntime": "1.14.0", "onnxruntime-gpu": "1.14.0"},
        {}
    ]

    ok = 0
    for c in cases:
        if check_shadow_wheel(c) == ref.check_shadow_wheel(c):
            ok += 1

    if ok == len(cases):
        out["cases_matched"] = 1.0

    return out
