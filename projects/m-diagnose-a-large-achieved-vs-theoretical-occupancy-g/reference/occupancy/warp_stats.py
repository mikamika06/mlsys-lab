import numpy as np

def diagnose_gap(theoretical_occ, achieved_occ, warp_stats):
    gap = theoretical_occ - achieved_occ
    if gap < 0.05:
        return "optimal"

    dominant_stall = max(warp_stats, key=warp_stats.get)
    if dominant_stall in ("stall_not_selected", "sched_constrained"):
        return "scheduler_bottleneck"
    elif dominant_stall in ("stall_mio_throttle", "stall_lg_throttle", "stall_mem_throttle"):
        return "memory_latency_throttle"
    elif dominant_stall in ("stall_drain", "stall_barrier"):
        return "synchronization_stall"
    else:
        return "register_dependency_stall"
