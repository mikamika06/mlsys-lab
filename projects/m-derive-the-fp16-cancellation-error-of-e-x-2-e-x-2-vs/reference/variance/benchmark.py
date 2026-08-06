def analyze_bandwidth_advantage(records):
    unfused = [r for r in records if r["kernel"] == "unfused"]
    fused = [r for r in records if r["kernel"] == "fused"]
    b_unfused = sum(r["bytes_transferred"] / r["time_ms"] for r in unfused)
    b_fused = sum(r["bytes_transferred"] / r["time_ms"] for r in fused)
    return float(b_fused / b_unfused if b_unfused > 0 else 1.0)
