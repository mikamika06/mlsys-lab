import numpy as np

def get_test_cases():
    return [
        {"seq_len": 4096, "layers": 32, "kv_heads": 8, "head_dim": 128, "sink": 4, "window": 512},
        {"seq_len": 16384, "layers": 32, "kv_heads": 8, "head_dim": 128, "sink": 4, "window": 1024},
        {"seq_len": 32768, "layers": 32, "kv_heads": 8, "head_dim": 128, "sink": 8, "window": 2048},
        {"seq_len": 65536, "layers": 32, "kv_heads": 8, "head_dim": 128, "sink": 16, "window": 4096},
    ]

def get_perplexity_data():
    seq_lens = [1000, 5000, 10000, 20000, 30000, 40000]
    full_ppl = [4.5, 4.6, 4.5, 4.6, 4.5, 4.6]
    sink_ppl = [4.5, 4.7, 4.8, 4.8, 4.9, 4.9]
    random_ppl = [4.5, 8.2, 14.5, 22.1, 35.4, 45.0]
    return seq_lens, full_ppl, sink_ppl, random_ppl

def compute_kv_bytes(seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    return 2 * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_elem

def compute_sink_window_bytes(seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int, sink_size: int, window_size: int, bytes_per_elem: int = 2) -> int:
    active_len = min(seq_len, sink_size + window_size)
    return 2 * active_len * num_layers * num_kv_heads * head_dim * bytes_per_elem

def evaluate_perplexity_curves(seq_lens: list, full_ppl: list, sink_ppl: list, random_ppl: list) -> dict:
    f_arr = np.array(full_ppl, dtype=float)
    s_arr = np.array(sink_ppl, dtype=float)
    r_arr = np.array(random_ppl, dtype=float)

    sink_better_count = int(np.sum(s_arr <= r_arr + 1e-5))
    random_spike_detected = bool(np.max(r_arr) > np.mean(f_arr) * 1.5)

    return {
        "sink_better_count": sink_better_count,
        "random_spike_detected": random_spike_detected,
        "mean_sink_ppl": float(np.mean(s_arr)),
        "mean_random_ppl": float(np.mean(r_arr))
    }
