def build_block_timing_table(grid, base_time):
    table = []
    gx, gy, gz = grid
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                exec_time = base_time * (1.0 + 0.1 * ((x + y + z) % 3))
                table.append({"block_id": [x, y, z], "duration_us": float(exec_time)})
    return table
