import numpy as np
from occupancy.resource import rank_kernels, compute_theoretical_occupancy
from occupancy.warp_stats import diagnose_gap

LIMITS = {
    "max_threads_per_sm": 2048,
    "max_regs_per_sm": 65536,
    "max_smem_sm": 102400,
    "max_blocks_per_sm": 32,
    "reg_allocation_unit": 256,
    "thread_allocation_unit": 32
}

CONFIGS = [
    {"id": "kernel_a", "regs_per_thread": 32, "threads_per_block": 256, "smem_per_block": 16384},
    {"id": "kernel_b", "regs_per_thread": 64, "threads_per_block": 128, "smem_per_block": 32768},
    {"id": "kernel_c", "regs_per_thread": 16, "threads_per_block": 512, "smem_per_block": 8192},
    {"id": "kernel_d", "regs_per_thread": 128, "threads_per_block": 64, "smem_per_block": 49152}
]

WARP_STATS_SAMPLES = [
    {"stall_not_selected": 45.0, "stall_mio_throttle": 10.0, "stall_drain": 5.0},
    {"stall_mio_throttle": 60.0, "stall_not_selected": 20.0, "stall_drain": 5.0},
    {"stall_drain": 55.0, "stall_not_selected": 15.0, "stall_mio_throttle": 10.0}
]
