import math
import numpy as np

PREFIX_TOKENS = np.array([101, 202, 303, 404, 505], dtype=np.int64)

TEST_PROMPTS = [
    np.array([1, 2, 3], dtype=np.int64),
    np.array([4, 5, 6, 7], dtype=np.int64),
    np.array([8], dtype=np.int64),
]

BENCHMARK_CONFIGS = [
    {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "batch_size": 1,
        "max_seq_len": 4096,
        "current_seq_len": 1024,
        "dtype_bytes": 2,
        "quant_bits": 8,
        "offload_gpu_fraction": 0.25,
    },
    {
        "num_layers": 16,
        "num_kv_heads": 4,
        "head_dim": 64,
        "batch_size": 4,
        "max_seq_len": 2048,
        "current_seq_len": 512,
        "dtype_bytes": 2,
        "quant_bits": 4,
        "offload_gpu_fraction": 0.1,
    },
    {
        "num_layers": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "batch_size": 2,
        "max_seq_len": 8192,
        "current_seq_len": 2048,
        "dtype_bytes": 4,
        "quant_bits": 8,
        "offload_gpu_fraction": 0.2,
    },
]


class DynamicCache:
    """Reference DynamicCache implementation."""

    def __init__(self, key_cache=None, value_cache=None):
        self.key_cache = list(key_cache) if key_cache is not None else []
        self.value_cache = list(value_cache) if value_cache is not None else []

    def update(self, key_states, value_states, layer_idx):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states.copy()
            self.value_cache[layer_idx] = value_states.copy()
        else:
            self.key_cache[layer_idx] = np.concatenate(
                [self.key_cache[layer_idx], key_states], axis=2
            )
            self.value_cache[layer_idx] = np.concatenate(
                [self.value_cache[layer_idx], value_states], axis=2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[2]
        return 0

    def crop(self, max_length):
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None:
                cur_len = self.key_cache[i].shape[2]
                if cur_len > max_length:
                    self.key_cache[i] = self.key_cache[i][:, :, :max_length, :]
                    self.value_cache[i] = self.value_cache[i][:, :, :max_length, :]

    def slice(self, start, end):
        if start < 0 or (end is not None and end < start):
            raise ValueError("Invalid slice bounds")
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None:
                self.key_cache[i] = self.key_cache[i][:, :, start:end, :]
                self.value_cache[i] = self.value_cache[i][:, :, start:end, :]

    def copy(self):
        new_k = [k.copy() if k is not None else None for k in self.key_cache]
        new_v = [v.copy() if v is not None else None for v in self.value_cache]
        return DynamicCache(key_cache=new_k, value_cache=new_v)


class SimulatedModel:
    """Simulated Transformer model for testing KV cache interaction."""

    def __init__(self, num_layers=2, num_heads=4, head_dim=16, vocab_size=100):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size

    def forward(self, tokens, past_key_values=None, start_pos=0):
        tokens_arr = np.asarray(tokens)
        if tokens_arr.ndim == 1:
            seq_len = tokens_arr.shape[0]
        else:
            seq_len = tokens_arr.shape[1]

        if past_key_values is None:
            past_key_values = DynamicCache()

        for layer_idx in range(self.num_layers):
            vals = tokens_arr.reshape(1, 1, seq_len, 1).astype(np.float32)
            k_states = (
                np.ones((1, self.num_heads, seq_len, self.head_dim), dtype=np.float32)
                * vals
            )
            v_states = (
                np.ones((1, self.num_heads, seq_len, self.head_dim), dtype=np.float32)
                * (vals + 1.0)
            )
            past_key_values.update(k_states, v_states, layer_idx)

        logits = np.zeros((1, seq_len, self.vocab_size), dtype=np.float32)
        return logits, past_key_values


def prime_prefix(prefix_tokens, model_fn):
    cache = DynamicCache()
    logits, cache = model_fn(prefix_tokens, past_key_values=cache, start_pos=0)
    return cache, logits


def reuse_prefix_priming(prefix_cache, prompt_tokens_list, model_fn):
    results = []
    for prompt in prompt_tokens_list:
        cloned = prefix_cache.copy()
        start_pos = cloned.get_seq_length()
        logits, updated_cache = model_fn(
            prompt, past_key_values=cloned, start_pos=start_pos
        )
        results.append((logits, updated_cache))
    return results


def benchmark_cache_implementations(
    num_layers,
    num_kv_heads,
    head_dim,
    batch_size,
    max_seq_len,
    current_seq_len,
    dtype_bytes=2,
    quant_bits=8,
    offload_gpu_fraction=0.2,
):
    base_elements_per_token = 2 * num_layers * batch_size * num_kv_heads * head_dim
    static_bytes = base_elements_per_token * max_seq_len * dtype_bytes
    static_peak = static_bytes

    dynamic_bytes = base_elements_per_token * current_seq_len * dtype_bytes
    dynamic_peak = static_bytes

    offloaded_bytes = int(math.ceil(static_bytes * offload_gpu_fraction))
    offloaded_peak = offloaded_bytes

    quant_elem_bytes = quant_bits / 8.0
    quantized_bytes = int(
        math.ceil(base_elements_per_token * current_seq_len * quant_elem_bytes)
    )
    quantized_peak = int(
        math.ceil(base_elements_per_token * max_seq_len * quant_elem_bytes)
    )

    def calc_savings(alloc):
        if static_bytes == 0:
            return 0.0
        return float(1.0 - (alloc / static_bytes))

    return {
        "dynamic": {
            "allocated_bytes": dynamic_bytes,
            "peak_bytes": dynamic_peak,
            "memory_savings_vs_static": calc_savings(dynamic_bytes),
            "supports_dynamic_growth": True,
        },
        "static": {
            "allocated_bytes": static_bytes,
            "peak_bytes": static_peak,
            "memory_savings_vs_static": 0.0,
            "supports_dynamic_growth": False,
        },
        "offloaded": {
            "allocated_bytes": offloaded_bytes,
            "peak_bytes": offloaded_peak,
            "memory_savings_vs_static": calc_savings(offloaded_bytes),
            "supports_dynamic_growth": False,
        },
        "quantized": {
            "allocated_bytes": quantized_bytes,
            "peak_bytes": quantized_peak,
            "memory_savings_vs_static": calc_savings(quantized_bytes),
            "supports_dynamic_growth": True,
        },
    }
