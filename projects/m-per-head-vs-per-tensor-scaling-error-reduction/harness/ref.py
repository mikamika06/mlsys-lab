import numpy as np


def generate_fixture():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=(64, 8, 32))
    x[:, 3, :] *= 1000.0  # Head 3 has an extreme outlier
    return x


def simulate_e4m3(x, scale):
    q = np.clip(np.round(x * scale), -448.0, 448.0)
    return q / scale


def get_per_tensor_scale(x, max_val=448.0):
    m = np.max(np.abs(x))
    return float(max_val / m) if m > 0 else 1.0


def get_per_head_scale(x, max_val=448.0):
    m = np.max(np.abs(x), axis=(0, 2), keepdims=True)
    return np.where(m > 0, max_val / m, 1.0)


def measure_rel_err(orig, approx):
    denom = np.mean(np.abs(orig))
    if denom == 0:
        return 0.0
    return float(np.mean(np.abs(orig - approx)) / denom)


def find_breaking_head(x, max_val=448.0):
    st = get_per_tensor_scale(x, max_val)
    sh = get_per_head_scale(x, max_val)
    qt = simulate_e4m3(x, st)
    qh = simulate_e4m3(x, sh)
    worst_diff = -1.0
    worst_head = -1
    for i in range(x.shape[1]):
        diff = measure_rel_err(x[:, i, :], qt[:, i, :]) - measure_rel_err(x[:, i, :], qh[:, i, :])
        if diff > worst_diff:
            worst_diff = diff
            worst_head = i
    return worst_head


def calc_cache_bytes(seq_len, num_heads, head_dim, num_layers, policy):
    elements = seq_len * num_heads * head_dim * num_layers
    if policy == "fp16":
        return elements * 2
    if policy == "fp8_per_tensor":
        return elements + (num_layers * 4)
    if policy == "fp8_per_head":
        return elements + (num_layers * num_heads * 4)
    raise ValueError("Unknown policy")
