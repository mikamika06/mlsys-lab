CASES = [
    {"seq_len": 50, "num_sinks": 2, "window_size": 10},
    {"seq_len": 5, "num_sinks": 4, "window_size": 10},
    {"seq_len": 100, "num_sinks": 4, "window_size": 20},
]

MATRICES = [
    [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],
    [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
]

def streaming_llm_indices(seq_len, num_sinks, window_size):
    if seq_len <= 0:
        return []
    sinks = list(range(min(seq_len, num_sinks)))
    if seq_len <= num_sinks:
        return sorted(list(set(sinks)))
    start = max(num_sinks, seq_len - window_size)
    window = list(range(start, seq_len))
    return sorted(list(set(sinks + window)))

def h2o_heavy_hitters(attn_matrix, capacity):
    import numpy as np
    arr = np.array(attn_matrix, dtype=float)
    if arr.size == 0 or capacity <= 0:
        return []
    scores = arr.sum(axis=0)
    if len(scores) <= capacity:
        return sorted(list(range(len(scores))))
    indices = np.argsort(scores)[-capacity:]
    return sorted(indices.tolist())

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
