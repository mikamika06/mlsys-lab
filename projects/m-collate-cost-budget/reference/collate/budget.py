"""Collate profiling and budget verification utilities."""

def profile_collate(collate_fn, sample_generator, batch_sizes):
    """Profile collate_fn execution time across various batch sizes."""
    results = {}
    for bs in sorted(batch_sizes):
        samples = sample_generator(bs)
        start_time, end_time = collate_fn(samples)
        total_time_ms = (end_time - start_time) * 1000.0
        per_sample_ms = total_time_ms / float(bs) if bs > 0 else 0.0
        results[bs] = {
            "batch_size": bs,
            "total_time_ms": total_time_ms,
            "per_sample_ms": per_sample_ms,
        }
    return results


def evaluate_budget(profile_results, max_budget_ms_per_batch, target_throughput_samples_sec):
    """Evaluate whether the collate profile satisfies budget and throughput rules."""
    evaluations = {}
    for bs, stats in profile_results.items():
        time_ms = stats["total_time_ms"]
        throughput = (bs / (time_ms / 1000.0)) if time_ms > 0 else 0.0

        within_budget = time_ms <= max_budget_ms_per_batch
        meets_throughput = throughput >= target_throughput_samples_sec

        evaluations[bs] = {
            "within_budget": within_budget,
            "meets_throughput": meets_throughput,
            "actual_time_ms": time_ms,
            "actual_throughput": throughput,
            "compliant": within_budget and meets_throughput,
        }
    return evaluations
