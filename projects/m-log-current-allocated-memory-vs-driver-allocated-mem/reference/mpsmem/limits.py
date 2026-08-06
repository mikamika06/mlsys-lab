from mpsmem.allocator import SimulatedMPSAllocator


def find_empirical_oom_threshold(capacity_bytes: int, recommended_max_bytes: int, workload: list) -> dict:
    allocator_no_cleanup = SimulatedMPSAllocator(capacity_bytes)
    res_no_cleanup = allocator_no_cleanup.run_workload(workload)

    workload_with_cleanup = []
    for op in workload:
        workload_with_cleanup.append(op)
        if op[0] == "free":
            workload_with_cleanup.append(("empty_cache",))

    allocator_with_cleanup = SimulatedMPSAllocator(capacity_bytes)
    res_with_cleanup = allocator_with_cleanup.run_workload(workload_with_cleanup)

    return {
        "recommended_max_bytes": recommended_max_bytes,
        "capacity_bytes": capacity_bytes,
        "without_cleanup": res_no_cleanup,
        "with_cleanup": res_with_cleanup,
        "exceeds_recommended_without_cleanup": res_no_cleanup["driver_allocated"] > recommended_max_bytes,
        "exceeds_recommended_with_cleanup": res_with_cleanup["driver_allocated"] > recommended_max_bytes,
    }
