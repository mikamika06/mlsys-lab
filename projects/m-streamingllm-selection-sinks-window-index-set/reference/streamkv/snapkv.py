def snapkv_pool_scores(attn_matrix, window_size, cluster_size):
    import numpy as np
    arr = np.array(attn_matrix, dtype=float)
    if arr.size == 0 or window_size <= 0 or cluster_size <= 0:
        return []
    seq_len = arr.shape[1]
    obs_start = max(0, seq_len - window_size)
    obs_matrix = arr[:, obs_start:]
    if obs_matrix.shape[1] == 0:
        return []
    num_cols = obs_matrix.shape[1]
    pooled = []
    for i in range(0, num_cols, cluster_size):
        chunk = obs_matrix[:, i:i+cluster_size]
        pooled.append(float(chunk.sum()))
    best_chunk_idx = int(np.argmax(pooled))
    start_idx = obs_start + best_chunk_idx * cluster_size
    end_idx = min(seq_len, start_idx + cluster_size)
    return list(range(start_idx, end_idx))
