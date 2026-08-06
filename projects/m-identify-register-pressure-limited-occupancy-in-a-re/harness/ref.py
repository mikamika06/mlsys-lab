import numpy as np

NCU_PROFILES = [
    {
        "regs_per_thread": 32,
        "shmem_per_block": 1024,
        "block_size": 256,
        "device_specs": {"max_regs_per_sm": 65536, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "warp_size": 32, "reg_allocation_unit": 256, "shmem_allocation_unit": 1024}
    },
    {
        "regs_per_thread": 64,
        "shmem_per_block": 16384,
        "block_size": 512,
        "device_specs": {"max_regs_per_sm": 65536, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "warp_size": 32, "reg_allocation_unit": 256, "shmem_allocation_unit": 1024}
    },
    {
        "regs_per_thread": 128,
        "shmem_per_block": 32768,
        "block_size": 1024,
        "device_specs": {"max_regs_per_sm": 65536, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "warp_size": 32, "reg_allocation_unit": 256, "shmem_allocation_unit": 1024}
    }
]

def compute_occupancy(regs_per_thread, shmem_per_block, block_size, device_specs):
    warp_size = device_specs["warp_size"]
    warps_per_block = (block_size + warp_size - 1) // warp_size
    reg_unit = device_specs["reg_allocation_unit"]
    regs_per_warp = ((regs_per_thread * warp_size + reg_unit - 1) // reg_unit) * reg_unit
    total_regs_per_block = regs_per_warp * warps_per_block
    max_blocks_by_regs = device_specs["max_regs_per_sm"] // total_regs_per_block if total_regs_per_block > 0 else 0
    shmem_unit = device_specs["shmem_allocation_unit"]
    allocated_shmem = ((shmem_per_block + shmem_unit - 1) // shmem_unit) * shmem_unit
    max_blocks_by_shmem = 32768 // allocated_shmem if allocated_shmem > 0 else 32
    max_threads_per_sm = device_specs["max_threads_per_sm"]
    max_blocks_by_threads = max_threads_per_sm // block_size
    max_blocks_by_sm_limit = device_specs["max_blocks_per_sm"]
    active_blocks = min(max_blocks_by_regs, max_blocks_by_shmem, max_blocks_by_threads, max_blocks_by_sm_limit)
    active_threads = active_blocks * block_size
    max_possible_threads = max_threads_per_sm
    occupancy = float(active_threads) / float(max_possible_threads)
    limiting_factor = "registers"
    if active_blocks == max_blocks_by_shmem:
        limiting_factor = "shared_memory"
    elif active_blocks == max_blocks_by_threads:
        limiting_factor = "threads"
    elif active_blocks == max_blocks_by_sm_limit:
        limiting_factor = "sm_blocks"
    return {"occupancy": float(occupancy), "limiting_factor": limiting_factor, "active_blocks": int(active_blocks)}

GRID_TESTS = [
    {"grid": (4, 1, 1), "base_time": 10.0},
    {"grid": (8, 2, 1), "base_time": 5.0},
    {"grid": (16, 1, 1), "base_time": 2.5}
]

def build_block_timing_table(grid, base_time):
    table = []
    gx, gy, gz = grid
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                exec_time = base_time * (1.0 + 0.1 * ((x + y + z) % 3))
                table.append({"block_id": [x, y, z], "duration_us": float(exec_time)})
    return table
