import ref


def check(workdir):
    from minja_tools.filters import UnsupportedFilterDetector
    detector = UnsupportedFilterDetector(ref.SUPPORTED)
    ok = 0
    for cfg in ref.CONFIGS:
        got = sorted(detector.find_unsupported(cfg["template"]))
        want = sorted(cfg["unsupported"])
        if got == want:
            ok += 1
    return {"filters_matched": float(ok)}
