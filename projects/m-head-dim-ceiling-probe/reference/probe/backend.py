def compare_fa_throughput(fa2_metrics, fa3_metrics):
    fa2_tps = [m["tokens_per_sec"] for m in fa2_metrics]
    fa3_tps = [m["tokens_per_sec"] for m in fa3_metrics]
    ratio = sum(fa3_tps) / (sum(fa2_tps) + 1e-9)
    return {"fa2_mean": sum(fa2_tps) / len(fa2_tps), "fa3_mean": sum(fa3_tps) / len(fa3_tps), "ratio": ratio}
