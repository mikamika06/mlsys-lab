import numpy as np
from kvcache.config import LayerConfig, ModelConfig


def make_sample_config(num_layers=4, num_heads=2, head_dim=16, window_size=8):
    layer_configs = []
    for i in range(num_layers):
        is_sliding = (i % 2 == 1)
        ws = window_size if is_sliding else None
        layer_configs.append(LayerConfig(layer_id=i, is_sliding=is_sliding, window_size=ws))
    return ModelConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        layer_configs=layer_configs,
    )


def generate_synthetic_data(seq_len, num_heads, head_dim, seed=42):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((seq_len, num_heads, head_dim), dtype=np.float32)
    k = rng.standard_normal((seq_len, num_heads, head_dim), dtype=np.float32)
    v = rng.standard_normal((seq_len, num_heads, head_dim), dtype=np.float32)
    return q, k, v


class FullKVCache:

    def __init__(self, num_layers: int):
        self.k_list = [None] * num_layers
        self.v_list = [None] * num_layers

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray):
        if self.k_list[layer_idx] is None:
            self.k_list[layer_idx] = k.copy()
            self.v_list[layer_idx] = v.copy()
        else:
            self.k_list[layer_idx] = np.concatenate(
                [self.k_list[layer_idx], k], axis=0
            )
            self.v_list[layer_idx] = np.concatenate(
                [self.v_list[layer_idx], v], axis=0
            )
        return self.k_list[layer_idx], self.v_list[layer_idx]


class FullAttentionEngine:

    def __init__(self, config: ModelConfig, cache: FullKVCache):
        self.config = config
        self.cache = cache

    def forward_layer_step(
        self, layer_idx: int, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        lc = self.config.layer_configs[layer_idx]
        k_full, v_full = self.cache.update(layer_idx, k, v)
        seq_len = k_full.shape[0]

        q_t = np.transpose(q, (1, 0, 2))
        k_t = np.transpose(k_full, (1, 0, 2))
        v_t = np.transpose(v_full, (1, 0, 2))

        d_k = float(self.config.head_dim)
        scores = np.matmul(q_t, np.transpose(k_t, (0, 2, 1))) / np.sqrt(d_k)

        if lc.is_sliding and lc.window_size is not None:
            window = lc.window_size
            if seq_len > window:
                mask = np.zeros((1, seq_len), dtype=bool)
                mask[0, : seq_len - window] = True
                scores[:, :, mask[0]] = -1e9

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
            out_t = self.forward_layer_step(
                layer_idx, q_seq[t : t + 1], k_seq[t : t + 1], v_seq[t : t + 1]
            )
            outputs.append(out_t)
        return np.concatenate(outputs, axis=0)
