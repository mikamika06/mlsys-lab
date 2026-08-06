def simulate_traffic(strategy, mesh_shape=(2, 2)):
    rows, cols = mesh_shape
    ranks = rows * cols
    if strategy == "HYBRID_SHARD":
        inter = 1024 * 1024 * (cols - 1) * 2
        intra = 512 * 1024 * (rows - 1)
        total = inter + intra
        max_load = inter
    elif "FULL_SHARD" in strategy:
        total = 2048 * 1024 * (ranks - 1)
        max_load = total // cols
    else:
        total = 512 * 1024 * ranks
        max_load = total // rows
    return {
        "total_traffic": int(total),
        "max_link_load": int(max_load),
        "ranks": ranks
    }
