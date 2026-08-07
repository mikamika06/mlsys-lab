def classify_kernel_profile(warp_state_stats):
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
