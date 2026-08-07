import sys

def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from roofline.workload import classify_workload_dominance
    from roofline.telemetry import extract_achieved_hbm_bandwidth

    out = {"workloads_classified": 0.0, "bandwidth_matched": 0.0}

    cfg = ref.MODEL_CONFIGS[0]
    hw = ref.HARDWARE_SPECS[0]

    want_class = ref.classify_workload_dominance(ref.WORKLOAD_PROFILES, cfg, hw)
    got_class = classify_workload_dominance(ref.WORKLOAD_PROFILES, cfg, hw)

    class_ok = 0
    for g, w in zip(got_class, want_class):
        if g == w:
            class_ok += 1
        elif "_note" not in out:
            out["_note"] = f"classification mismatch: got {g}, want {w}"

    out["workloads_classified"] = float(class_ok)

    bw_ok = 0
    for metrics in ref.MEASURED_METRICS:
        want_bw = ref.extract_achieved_hbm_bandwidth(metrics, cfg)
        got_bw = extract_achieved_hbm_bandwidth(metrics, cfg)
        if abs(got_bw - want_bw) < 1e-4:
            bw_ok += 1
        elif "_note" not in out:
            out["_note"] = f"bandwidth mismatch: got {got_bw}, want {want_bw}"

    out["bandwidth_matched"] = float(bw_ok)
    return out
