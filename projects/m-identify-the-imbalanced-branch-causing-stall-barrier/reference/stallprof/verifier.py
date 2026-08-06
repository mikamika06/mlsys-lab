"""Verifier module for synchronization removal and stall drops."""

def verify_sync_removal(baseline_stats, modified_stats, speedup_ratio):
    base_barrier = baseline_stats["warp_stats"]["stall_barrier"]
    mod_barrier = modified_stats["warp_stats"]["stall_barrier"]
    drop_ratio = (base_barrier - mod_barrier) / base_barrier
    return drop_ratio >= 0.2 and speedup_ratio >= 1.15
