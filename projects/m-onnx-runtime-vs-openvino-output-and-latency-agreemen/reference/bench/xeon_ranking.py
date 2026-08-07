def rank_xeon_throughput(xeon_logs):
    """Computes median throughput (qps) for each tool and returns sorted ranking."""
    stats = {}
    for entry in xeon_logs:
        engine = entry["engine"]
        qps = entry["queries_per_sec"]
        stats.setdefault(engine, []).append(qps)

    rankings = []
    for engine, qps_list in stats.items():
        sorted_qps = sorted(qps_list)
        n = len(sorted_qps)
        mid = n // 2
        med = sorted_qps[mid] if n % 2 != 0 else (sorted_qps[mid - 1] + sorted_qps[mid]) / 2.0
        rankings.append({"engine": engine, "median_qps": float(med)})

    rankings.sort(key=lambda x: x["median_qps"], reverse=True)
    return rankings


def check_precision_fairness(engine_a_info, engine_b_info):
    """Verifies precision-matched fairness check between two CPU inference benchmarks."""
    prec_a = engine_a_info.get("precision")
    prec_b = engine_b_info.get("precision")
    quant_a = engine_a_info.get("quantized", False)
    quant_b = engine_b_info.get("quantized", False)

    is_fair = (prec_a == prec_b) and (quant_a == quant_b)
    return {
        "fair": is_fair,
        "engine_a_prec": prec_a,
        "engine_b_prec": prec_b,
    }
