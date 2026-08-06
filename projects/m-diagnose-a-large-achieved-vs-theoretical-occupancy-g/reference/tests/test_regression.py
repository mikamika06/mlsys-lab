from occupancy.warp_stats import diagnose_gap
from occupancy.resource import compute_theoretical_occupancy

def test_occupancy_regression():
    limits = {
        "max_threads_per_sm": 2048,
        "max_regs_per_sm": 65536,
        "max_smem_sm": 102400,
        "max_blocks_per_sm": 32,
        "reg_allocation_unit": 256,
        "thread_allocation_unit": 32
    }
    cfg = {"regs_per_thread": 32, "threads_per_block": 256, "smem_per_block": 16384}
    occ = compute_theoretical_occupancy(cfg, limits)
    assert occ > 0.0
    stats = {"stall_mio_throttle": 70.0, "stall_not_selected": 30.0}
    diag = diagnose_gap(occ, 0.2, stats)
    assert diag == "memory_latency_throttle"
