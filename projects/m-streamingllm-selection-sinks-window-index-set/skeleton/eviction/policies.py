import numpy as np


def select_streamingllm(seq_len: int, num_sinks: int, window_size: int) -> np.ndarray:
    """Select token indices under StreamingLLM sink and sliding window policy."""
    raise NotImplementedError


def select_h2o(attn_scores: np.ndarray, num_heavy_hitters: int, recent_window: int) -> np.ndarray:
    """Select token indices per head using Heavy-Hitter Oracle (H2O)."""
    raise NotImplementedError


def select_snapkv(attn_weights: np.ndarray, observation_window: int, max_capacity: int) -> np.ndarray:
    """Select token indices per head using SnapKV pooled observation scoring."""
    raise NotImplementedError
