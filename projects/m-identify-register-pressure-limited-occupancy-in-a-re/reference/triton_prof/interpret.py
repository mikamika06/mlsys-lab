def build_block_table(grid_shape, block_durations):
    """Build per-block timing table for TRITON_INTERPRET=1 runs."""
    table = []
    for i, duration in enumerate(block_durations):
        table.append({"block_id": i, "duration": float(duration), "status": "completed"})
    return table
