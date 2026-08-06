import os
import time

def build_block_timing_table(kernel_func, grid, *args, **kwargs):
    os.environ["TRITON_INTERPRET"] = "1"
    rows = []

    if len(grid) == 1:
        gx = grid[0]
        gy, gz = 1, 1
    elif len(grid) == 2:
        gx, gy = grid
        gz = 1
    else:
        gx, gy, gz = grid

    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                t0 = time.perf_counter_ns()
                try:
                    kernel_func[(x, y, z)](*args, **kwargs)
                    success = True
                except Exception:
                    success = False
                t1 = time.perf_counter_ns()
                duration_us = (t1 - t0) / 1000.0
                rows.append({
                    "block_idx": (x, y, z),
                    "duration_us": duration_us,
                    "success": success
                })
    return rows
