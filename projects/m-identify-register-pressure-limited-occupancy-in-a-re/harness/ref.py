import numpy as np

NCU_CASES = [
    {"regs_per_thread": 64, "threads_per_block": 256, "smem_bytes": 1024, "max_regs_per_thread": 255, "warp_size": 32, "expected_limit": "register"},
    {"regs_per_thread": 16, "threads_per_block": 1024, "smem_bytes": 49152, "max_regs_per_thread": 255, "warp_size": 32, "expected_limit": "smem"},
    {"regs_per_thread": 32, "threads_per_block": 128, "smem_bytes": 1024, "max_regs_per_thread": 255, "warp_size": 32, "expected_limit": "threads"},
]

def analyze_occupancy(profile_data):
    regs = profile_data["regs_per_thread"]
    threads = profile_data["threads_per_block"]
    warp_size = profile_data["warp_size"]
    warps = (threads + warp_size - 1) // warp_size
    reg_limit_blocks = 65536 // (regs * warps * warp_size) if regs > 0 else 32
    smem_limit = 49152 // max(1, profile_data["smem_bytes"])
    thread_limit = 2048 // threads
    limits = {"register": reg_limit_blocks, "smem": smem_limit, "threads": thread_limit}
    bottleneck = min(limits, key=limits.get)
    return {"bottleneck": bottleneck, "max_active_blocks": limits[bottleneck]}

def build_block_table(grid_shape, block_durations):
    table = []
    for i, duration in enumerate(block_durations):
        table.append({"block_id": i, "duration": float(duration), "status": "completed"})
    return table
