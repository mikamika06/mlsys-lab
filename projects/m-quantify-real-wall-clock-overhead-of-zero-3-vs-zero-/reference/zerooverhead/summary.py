from zerooverhead.overhead import calculate_zero3_overhead


def summarize_benchmark_runs(runs_data, warmup_steps=5):
    summary = []
    for run in runs_data:
        name = run.get("name", "unnamed")
        z2_recs = run.get("z2_records", [])
        z3_recs = run.get("z3_records", [])
        metrics = calculate_zero3_overhead(
            z2_recs, z3_recs, warmup_steps=warmup_steps
        )
        summary.append(
            {
                "name": name,
                "rel_overhead": metrics["rel_overhead"],
                "slowdown_ratio": metrics["slowdown_ratio"],
                "param_gather_pct": metrics["param_gather_pct"],
            }
        )
    return summary
