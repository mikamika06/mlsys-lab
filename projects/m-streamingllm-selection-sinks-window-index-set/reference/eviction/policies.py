import numpy as np


def select_streamingllm(seq_len: int, num_sinks: int, window_size: int) -> np.ndarray:
    """Select token indices under StreamingLLM sink and sliding window policy."""
    if seq_len <= num_sinks + window_size:
        return np.arange(seq_len, dtype=np.int64)
    sinks = np.arange(num_sinks, dtype=np.int64)
    recent = np.arange(seq_len - window_size, seq_len, dtype=np.int64)
    return np.concatenate([sinks, recent])


def select_h2o(attn_scores: np.ndarray, num_heavy_hitters: int, recent_window: int) -> np.ndarray:
    """Select token indices per head using Heavy-Hitter Oracle (H2O)."""
    num_heads, seq_len = attn_scores.shape
    if seq_len <= num_heavy_hitters + recent_window:
        return np.tile(np.arange(seq_len, dtype=np.int64), (num_heads, 1))

    recent_start = seq_len - recent_window
    candidate_scores = attn_scores[:, :recent_start]
    hh_count = min(num_heavy_hitters, recent_start)

    result = np.zeros((num_heads, hh_count + recent_window), dtype=np.int64)
    recent_indices = np.arange(recent_start, seq_len, dtype=np.int64)

    for h in range(num_heads):
        top_hh = np.argsort(candidate_scores[h])[-hh_count:]
        selected = np.sort(np.concatenate([top_hh, recent_indices]))
        result[h] = selected

    return result


def select_snapkv(attn_weights: np.ndarray, observation_window: int, max_capacity: int) -> np.ndarray:
    """Select token indices per head using SnapKV pooled observation scoring."""
    num_heads, num_queries, seq_len = attn_weights.shape
    if seq_len <= max_capacity:
        return np.tile(np.arange(seq_len, dtype=np.int64), (num_heads, 1))

    obs_len = min(observation_window, num_queries)
    obs_weights = attn_weights[:, -obs_len:, :]
    importance = np.mean(obs_weights, axis=1)

    recent_len = min(obs_len, max_capacity)
    prefix_len = seq_len - recent_len
    prefix_budget = max_capacity - recent_len

    result = np.zeros((num_heads, prefix_budget + recent_len), dtype=np.int64)
    recent_indices = np.arange(prefix_len, seq_len, dtype=np.int64)

    for h in range(num_heads):
        if prefix_budget > 0 and prefix_len > 0:
            top_prefix = np.argsort(importance[h, :prefix_len])[-prefix_budget:]
            selected = np.sort(np.concatenate([top_prefix, recent_indices]))
        else:
            selected = recent_indices
        result[h] = selected

    return result
