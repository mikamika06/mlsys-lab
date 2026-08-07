import ref

def check(workdir):
    from servemetrics.bench import compute_tpot_overhead
    from servemetrics.report import generate_report
    out = {"latency_ratio": 0.0, "report_valid": 0.0}
    cfg = ref.CONFIGS[1]
    ref_res = ref.compute_tpot_overhead(cfg, seed=42)
    got_res = compute_tpot_overhead(cfg, seed=42)
    if got_res and abs(got_res.get("latency_ratio", 0) - ref_res["latency_ratio"]) < 1e-4:
        out["latency_ratio"] = 1.0

    report_text = generate_report(cfg, seed=42)
    if report_text and isinstance(report_text, str) and len(report_text) > 10:
        out["report_valid"] = 1.0
    return out
