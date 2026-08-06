import ref

def check(workdir):
    out = {"detector_accuracy": 0.0, "fallback_causes_correct": 0.0}

    try:
        from kernel_attr.detector import QuadraticMemoryDetector
        from kernel_attr.fallback import FallbackDiagnostics
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    detector = QuadraticMemoryDetector()
    got_analysis = detector.analyze_allocations(ref.PROFILES)
    exp_analysis = ref.ref_analyze_allocations(ref.PROFILES)

    if len(got_analysis) == len(exp_analysis):
        det_ok = True
        for g, e in zip(got_analysis, exp_analysis):
            if g["op_id"] != e["op_id"] or g["is_quadratic"] != e["is_quadratic"]:
                det_ok = False
                break
        if det_ok:
            out["detector_accuracy"] = 1.0
        else:
            out["_note"] = f"Detector mismatch: got {got_analysis}, expected {exp_analysis}"

    diag = FallbackDiagnostics()
    diag_ok = True
    for cfg, exp_cause in ref.FALLBACK_CONFIGS:
        got_cause = diag.diagnose_fallback(cfg)
        if got_cause != exp_cause:
            diag_ok = False
            out["_note"] = f"Fallback mismatch for {cfg}: got {got_cause}, expected {exp_cause}"
            break

    if diag_ok:
        out["fallback_causes_correct"] = 1.0

    return out
