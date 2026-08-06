import ref

def check(workdir):
    from bench.stats import compute_repeats_for_claim
    from bench.compare import build_comparison_table
    out = {"repeats_valid": 0.0, "throughput_ratio": 0.0}

    lats = [50.0, 52.0, 48.0, 51.0, 49.0]
    n = compute_repeats_for_claim(lats, 0.05)
    if isinstance(n, int) and n >= 1:
        out["repeats_valid"] = 1.0

    table = build_comparison_table(ref.SAMPLE_RUNS_FR, ref.SAMPLE_RUNS_MC)
    ratio = table.get("throughput_ratio", 0.0)
    out["throughput_ratio"] = float(ratio)
    return out
