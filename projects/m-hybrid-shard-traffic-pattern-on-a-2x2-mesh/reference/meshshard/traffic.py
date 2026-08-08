def compute_traffic(mesh_shape, shard_config):
    rows, cols = mesh_shape
    total_ranks = rows * cols
    traffic = {}
    for r in range(total_ranks):
        row_idx, col_idx = divmod(r, cols)
        peers = []
        for other in range(total_ranks):
            o_row, o_col = divmod(other, cols)
            if o_row == row_idx:
                peers.append((other, "intra"))
            elif o_col == col_idx:
                peers.append((other, "inter"))
            else:
                peers.append((other, "cross"))
        traffic[r] = peers
    return traffic
