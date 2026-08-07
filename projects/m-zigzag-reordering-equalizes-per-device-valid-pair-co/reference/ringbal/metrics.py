def workload_imbalance(assignment: list[list[int]]) -> dict[str, float]:
    workloads = []
    for dev_blocks in assignment:
        workloads.append(sum(i + 1 for i in dev_blocks))

    mean_w = sum(workloads) / len(workloads)
    max_w = max(workloads)

    return {
        "mean": float(mean_w),
        "max": float(max_w),
        "rel_err": float((max_w - mean_w) / mean_w) if mean_w > 0 else 0.0
    }
