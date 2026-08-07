import random

CATEGORIES = ["memory_bound", "sync_bound", "math_pipe_throttled", "control_divergent"]

def generate_warp_stats_datasets():
    random.seed(42)
    datasets = []
    for _ in range(5):
        stats = [
            {"reason": "Stall Long Scoreboard", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
            {"reason": "Stall Short Scoreboard", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
            {"reason": "Stall Barrier", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
            {"reason": "Stall Membar", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
            {"reason": "Stall MIO Throttle", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
            {"reason": "Stall Math Pipe Throttle", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
            {"reason": "Stall Branch Exec Render", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
            {"reason": "Stall Selected", "total_stall_cycles": random.randint(1000, 50000), "total_executed_instructions": random.randint(100, 1000)},
        ]
        datasets.append(stats)
    return datasets

def generate_kernel_classification_cases():
    cases = [
        (
            [
                {"reason": "Stall Long Scoreboard", "total_stall_cycles": 80000, "total_executed_instructions": 1000},
                {"reason": "Stall Barrier", "total_stall_cycles": 1000, "total_executed_instructions": 1000},
                {"reason": "Stall Math Pipe Throttle", "total_stall_cycles": 2000, "total_executed_instructions": 1000},
                {"reason": "Stall Branch Exec Render", "total_stall_cycles": 500, "total_executed_instructions": 1000},
            ],
            "memory_bound"
        ),
        (
            [
                {"reason": "Stall Long Scoreboard", "total_stall_cycles": 1000, "total_executed_instructions": 1000},
                {"reason": "Stall Barrier", "total_stall_cycles": 40000, "total_executed_instructions": 1000},
                {"reason": "Stall Membar", "total_stall_cycles": 30000, "total_executed_instructions": 1000},
                {"reason": "Stall Math Pipe Throttle", "total_stall_cycles": 2000, "total_executed_instructions": 1000},
            ],
            "sync_bound"
        ),
        (
            [
                {"reason": "Stall Long Scoreboard", "total_stall_cycles": 2000, "total_executed_instructions": 1000},
                {"reason": "Stall Barrier", "total_stall_cycles": 1000, "total_executed_instructions": 1000},
                {"reason": "Stall MIO Throttle", "total_stall_cycles": 30000, "total_executed_instructions": 1000},
                {"reason": "Stall Math Pipe Throttle", "total_stall_cycles": 40000, "total_executed_instructions": 1000},
            ],
            "math_pipe_throttled"
        ),
        (
            [
                {"reason": "Stall Long Scoreboard", "total_stall_cycles": 1000, "total_executed_instructions": 1000},
                {"reason": "Stall Branch Exec Render", "total_stall_cycles": 35000, "total_executed_instructions": 1000},
                {"reason": "Stall Selected", "total_stall_cycles": 25000, "total_executed_instructions": 1000},
                {"reason": "Stall MIO Throttle", "total_stall_cycles": 2000, "total_executed_instructions": 1000},
            ],
            "control_divergent"
        )
    ]
    return cases

def generate_scheduler_stats_cases():
    random.seed(123)
    cases = []
    for _ in range(5):
        total_cycles = 10000.0
        issued = float(random.randint(4000, 16000))
        max_slots = 20000.0
        eligible = float(random.randint(1, 4))
        no_eligible = float(random.randint(1000, 5000))

        stats = {
            "issued_slots": issued,
            "max_issue_slots": max_slots,
            "eligible_warps_per_cycle": eligible,
            "no_eligible_warp_cycles": no_eligible,
            "total_cycles": total_cycles
        }
        cases.append(stats)
    return cases

def ref_top_stall_reasons(warp_state_stats, k=3):
    ranked = []
    for entry in warp_state_stats:
        reason = entry["reason"]
        total_cycles = float(entry["total_stall_cycles"])
        total_insts = float(entry["total_executed_instructions"])
        avg_cpi = total_cycles / total_insts if total_insts > 0 else 0.0
        ranked.append((avg_cpi, total_cycles, reason))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [{"reason": r, "avg_cpi": cpi} for cpi, _, r in ranked[:k]]

def ref_classify_kernel_profile(warp_state_stats):
    cpi_by_reason = {}
    for entry in warp_state_stats:
        reason = entry["reason"]
        insts = float(entry["total_executed_instructions"])
        cycles = float(entry["total_stall_cycles"])
        cpi = cycles / insts if insts > 0 else 0.0
        cpi_by_reason[reason] = cpi

    mem_cpi = cpi_by_reason.get("Stall Long Scoreboard", 0.0)
    sync_cpi = cpi_by_reason.get("Stall Barrier", 0.0) + cpi_by_reason.get("Stall Membar", 0.0)
    math_cpi = cpi_by_reason.get("Stall MIO Throttle", 0.0) + cpi_by_reason.get("Stall Math Pipe Throttle", 0.0)
    div_cpi = cpi_by_reason.get("Stall Branch Exec Render", 0.0) + cpi_by_reason.get("Stall Selected", 0.0)

    scores = {
        "memory_bound": mem_cpi,
        "sync_bound": sync_cpi,
        "math_pipe_throttled": math_cpi,
        "control_divergent": div_cpi
    }
    best = max(scores.items(), key=lambda x: x[1])
    return best[0]

def ref_compute_issue_slot_utilization(scheduler_stats):
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
