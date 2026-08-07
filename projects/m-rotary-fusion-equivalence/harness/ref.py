import numpy as np

def generate_test_inputs():
    np.random.seed(42)
    B, H, S, D = 1, 8, 128, 64
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    cos = np.random.randn(1, 1, S, D // 2).astype(np.float32)
    sin = np.random.randn(1, 1, S, D // 2).astype(np.float32)
    return q, k, cos, sin

def ref_apply_fused_rotary(q, k, cos, sin):
    def apply_one(x):
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return np.concatenate([out1, out2], axis=-1)

    return apply_one(q), apply_one(k)

def ref_optimal_num_splits(seq_len):
    if seq_len <= 512:
        return 1
    elif seq_len <= 2048:
        return 4
    else:
        return 8

def ref_decode_latency(cache_len):
    return float(0.1 + 0.0005 * cache_len)
