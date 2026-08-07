import numpy as np
from kvcache.cache import HybridKVCache
from kvcache.config import ModelConfig


class HybridAttentionEngine:

    def __init__(self, config: ModelConfig, cache: HybridKVCache):
        raise NotImplementedError

    def forward_layer_step(self, layer_idx: int, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def process_sequence(
        self,
        q_seq: np.ndarray,
        k_seq: np.ndarray,
        v_seq: np.ndarray,
        layer_idx: int,
    ) -> np.ndarray:
        raise NotImplementedError
