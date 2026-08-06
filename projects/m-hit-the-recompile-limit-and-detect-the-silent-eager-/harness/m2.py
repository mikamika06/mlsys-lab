import ref


def check(workdir):
    from recompile.detector import EagerFallbackDetector

    out = {"fallback_detected": 0.0}
    try:
        det = EagerFallbackDetector(limit=1)
        r1 = det.step(True)
        r2 = det.step(True)
        if not r1 and r2:
            out["fallback_detected"] = 1.0
        else:
            out["_note"] = f"detector steps returned {[r1, r2]}, expected [False, True]"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
