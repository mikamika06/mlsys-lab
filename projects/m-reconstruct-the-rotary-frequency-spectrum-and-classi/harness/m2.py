import ref

def check(workdir):
    from ropespectrum.spectrum import classify_dims
    out = {"classification_matched": 0.0}
    ok = 0
    threshold = 0.001
    for cfg in ref.CONFIGS:
        want = ref.classify_dims(cfg["dim"], cfg["base"], threshold)
        try:
            got = classify_dims(cfg["dim"], cfg["base"], threshold)
            if got == want:
                ok += 1
        except Exception:
            pass
    out["classification_matched"] = float(ok)
    return out
