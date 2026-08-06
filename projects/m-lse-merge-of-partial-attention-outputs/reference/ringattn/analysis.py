import numpy as np


def compute_causal_load_imbalance(world_size, num_blocks):
    total_blocks = world_size * num_blocks
    rank_work = np.zeros(world_size, dtype=np.int64)

    for r in range(world_size):
        q_start = r * num_blocks
        q_end = (r + 1) * num_blocks
        for b in range(total_blocks):
            if b < q_start:
                rank_work[r] += num_blocks
            elif b < q_end:
                n_full = b - q_start
                rank_work[r] += n_full

    max_work = float(np.max(rank_work))
    avg_work = float(np.mean(rank_work))
    ratio = max_work / max(avg_work, 1e-9)
    return {"rank_work": rank_work.tolist(), "imbalance_ratio": ratio}
