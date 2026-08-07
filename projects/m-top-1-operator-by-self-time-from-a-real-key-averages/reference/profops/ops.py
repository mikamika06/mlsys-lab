def top_1_operator_by_self_time(records):
    """Return name of operator with highest self cpu time."""
    if not records:
        return ""
    best = max(records, key=lambda r: r["self_cpu_time_total"])
    return best["name"]


def matmul_family_share(records):
    """Compute share of total self time belonging to matmul family."""
    if not records:
        return 0.0
    prefixes = ("aten::matmul", "aten::mm", "aten::bmm", "aten::addmm")
    total = sum(r["self_cpu_time_total"] for r in records)
    if total == 0:
        return 0.0
    mm_total = sum(r["self_cpu_time_total"] for r in records if r["name"].startswith(prefixes))
    return mm_total / total


def row_count_delta_across_batch_sizes(batch_tables):
    """Compute row count delta across sorted batch sizes."""
    counts = {b: len(recs) for b, recs in batch_tables.items()}
    sorted_batches = sorted(counts.keys())
    if len(sorted_batches) < 2:
        return 0
    deltas = [counts[sorted_batches[i+1]] - counts[sorted_batches[i]] for i in range(len(sorted_batches) - 1)]
    return sum(deltas)
