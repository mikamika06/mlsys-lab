def compute_issue_slot_utilization(scheduler_stats):
    issued = float(scheduler_stats["issued_slots"])
    max_slots = float(scheduler_stats["max_issue_slots"])
    eligible = float(scheduler_stats["eligible_warps_per_cycle"])
    no_eligible = float(scheduler_stats["no_eligible_warp_cycles"])
    total_cycles = float(scheduler_stats["total_cycles"])

    utilization = issued / max_slots if max_slots > 0 else 0.0
    issue_efficiency = issued / (eligible * total_cycles) if (eligible * total_cycles) > 0 else 0.0
    starvation_rate = no_eligible / total_cycles if total_cycles > 0 else 0.0

    return {
        "issue_slot_utilization": utilization,
        "issue_efficiency": issue_efficiency,
        "scheduler_starvation_rate": starvation_rate
    }
