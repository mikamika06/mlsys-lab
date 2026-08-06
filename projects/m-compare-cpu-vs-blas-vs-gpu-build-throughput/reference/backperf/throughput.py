def compare_throughput(records):
    results = {}
    for r in records:
        b = r["backend"]
        results[b] = {
            "build_time_s": float(r["build_time_s"]),
            "tokens_per_sec": float(r["tokens_per_sec"]),
            ("speedup_vs_cpu" if b != "cpu" else "baseline"): (
                float(r["tokens_per_sec"]) / float(records[0]["tokens_per_sec"])
                if b != "cpu" else 1.0
            )
        }
    return results
