import numpy as np
from kvcache.cache import HybridKVCache
from kvcache.config import ModelConfig


class HybridAttentionEngine:

    def __init__(self, config: ModelConfig, cache: HybridKVCache):
        self.config = config
        self.cache = cache

    def forward_layer_step(self, layer_idx: int, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        k_cached, v_cached = self.cache.update(layer_idx, k, v)

        q_t = np.transpose(q, (1, 0, 2))
        k_t = np.transpose(k_cached, (1, 0, 2))
        v_t = np.transpose(v_cached, (1, 0, 2))

        d_k = float(self.config.head_dim)
        scores = np.matmul(q_t, np.transpose(k_t, (0, 2, 1))) / np.sqrt(d_k)

        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        out_t = np.matmul(attn_weights, v_t)
        return np.transpose(out_t, (1, 0, 2))

    def process_sequence(
        self,
        q_seq: np.ndarray,
        k_seq: np.ndarray,
        v_seq: np.ndarray,
        layer_idx: int,
    ) -> np.ndarray:
        t_len = q_seq.shape[0]
        outputs = []
        for t in range(t_len):
            q_t = q_seq[t : t + 1]
            k_t = k_seq[t : t + 1]
            v_t = v_seq[t : t + 1]
            out_t = self.forward_layer_step(layer_idx, q_t, k_t, v_t)
            outputs.append(out_t)
        return np.concatenate(outputs, axis=0)
