def classify_kernels(reports: list[dict]) -> list[dict]:
    """Classify kernels from NCU metrics reports."""
    results = []
    for r in reports:
        sm = r.get("sm_throughput_pct", 0.0)
        dram = r.get("dram_throughput_pct", 0.0)
        achieved = r.get("achieved_occupancy_pct", 0.0)
        theoretical = r.get("theoretical_occupancy_pct", 0.0)

        if sm >= 60.0 and sm >= dram:
            bound = "compute"
        elif dram >= 60.0 and dram > sm:
            bound = "memory"
        elif theoretical > 0 and (achieved / theoretical) < 0.75:
            bound = "occupancy"
        else:
            bound = "unbound"

        results.append({"name": r["name"], "bound": bound})
    return results
