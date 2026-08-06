import ref

def check(workdir):
    from actmem.classify import classify_term
    out = {"classification_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.classify_dominating_term(cfg)
        got = classify_term(cfg)
        if got == want:
            ok += 1
    out["classification_matched"] = float(ok)
    return out
