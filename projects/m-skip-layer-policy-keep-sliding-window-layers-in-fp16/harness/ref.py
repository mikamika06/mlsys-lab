import numpy as np

def assign_kv_dtypes(layers):
    return ["float16" if l["kind"] == "sliding" else "float8" for l in layers]

def compute_kv_bytes(layers, dtypes, seq_len, batch_size, kv_heads, head_dim):
    total = 0
    for l, d in zip(layers, dtypes):
        bpe = 4 if d == "float32" else (2 if d == "float16" else 1)
        elements = min(seq_len, l["window"]) if l["kind"] == "sliding" else seq_len
        total += 2 * batch_size * kv_heads * head_dim * elements * bpe
    return total

def simulate_kv_cache_output(layers, dtypes, seq_len, hidden_dim, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(seq_len, hidden_dim)

    for layer, dtype in zip(layers, dtypes):
        k_w = rng.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        v_w = rng.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        o_w = rng.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)

        k = x @ k_w
        v = x @ v_w

        n_k = rng.randn(*k.shape)
        n_v = rng.randn(*v.shape)

        if dtype == "float8":
            k += n_k * k * 0.05
            v += n_v * v * 0.05
        elif dtype == "float16":
            k += n_k * k * 0.001
            v += n_v * v * 0.001

        attn = rng.randn(seq_len, seq_len)
        mask = np.tril(np.ones((seq_len, seq_len)))
        if layer["kind"] == "sliding":
            mask -= np.tril(np.ones((seq_len, seq_len)), -layer["window"])

        attn = attn * mask
        row_sums = attn.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        attn = attn / row_sums

        out = attn @ v
        x = x + out @ o_w

    return x

def eval_rel_err(layers, dtypes, seq_len, hidden_dim):
    ideal_dtypes = ["float32"] * len(layers)
    ideal = simulate_kv_cache_output(layers, ideal_dtypes, seq_len, hidden_dim)
    test = simulate_kv_cache_output(layers, dtypes, seq_len, hidden_dim)
    diff = np.linalg.norm(test - ideal)
    norm = np.linalg.norm(ideal)
    return float(diff / (norm + 1e-9))
