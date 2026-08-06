import numpy as np

def generate_size_fixtures():
    return [
        (
            {"w1": (4096, 4096), "w2": (4096, 1024)},
            {"w1_a": (16, 4096), "w1_b": (4096, 16)},
            2
        ),
        (
            {"attn": (1024, 1024)},
            {"attn_a": (8, 1024), "attn_b": (1024, 8)},
            4
        ),
        (
            {"proj": (512, 256)},
            {},
            2
        )
    ]

SIZE_FIXTURES = generate_size_fixtures()

def generate_num_fixtures():
    np.random.seed(42)
    fixtures = []
    for _ in range(5):
        in_dim = np.random.randint(64, 128)
        out_dim = np.random.randint(64, 128)
        r = np.random.randint(4, 16)
        seq = np.random.randint(10, 20)

        w = np.random.randn(out_dim, in_dim).astype(np.float32)
        a = np.random.randn(r, in_dim).astype(np.float32) * 0.1
        b = np.random.randn(out_dim, r).astype(np.float32) * 0.1
        scale = float(np.random.randint(1, 4))
        x = np.random.randn(seq, in_dim).astype(np.float32)
        fixtures.append((w, a, b, scale, x))
    return fixtures

NUMERICAL_FIXTURES = generate_num_fixtures()

def checkpoint_stats(base_shapes, lora_shapes, dtype_bytes=2):
    base_params = sum(np.prod(shape) for shape in base_shapes.values()) if base_shapes else 0
    lora_params = sum(np.prod(shape) for shape in lora_shapes.values()) if lora_shapes else 0

    base_bytes = int(base_params * dtype_bytes)
    lora_bytes = int(lora_params * dtype_bytes)
    ratio = float(lora_bytes / base_bytes) if base_bytes > 0 else 0.0

    return {
        "base_bytes": base_bytes,
        "lora_bytes": lora_bytes,
        "ratio": ratio
    }

def merge_weights(w_base, lora_a, lora_b, scale):
    w = w_base.astype(np.float32)
    a = lora_a.astype(np.float32)
    b = lora_b.astype(np.float32)
    return w + (b @ a) * scale

def quantization_error(w_base, lora_a, lora_b, scale):
    w_merged = merge_weights(w_base, lora_a, lora_b, scale)
    abs_max = float(np.max(np.abs(w_merged)))
    if abs_max == 0.0:
        return 0.0
    q_scale = abs_max / 127.0
    w_q = np.round(w_merged / q_scale) * q_scale
    diff = float(np.linalg.norm(w_merged - w_q))
    base = float(np.linalg.norm(w_merged))
    return diff / base if base > 0.0 else 0.0

def forward_equivalence(x, w_base, lora_a, lora_b, scale):
    x = x.astype(np.float32)
    w_base = w_base.astype(np.float32)
    lora_a = lora_a.astype(np.float32)
    lora_b = lora_b.astype(np.float32)

    y_adapter = (x @ w_base.T) + (x @ lora_a.T @ lora_b.T) * scale
    w_merged = merge_weights(w_base, lora_a, lora_b, scale)
    y_merged = x @ w_merged.T

    diff = float(np.linalg.norm(y_adapter - y_merged))
    base_norm = float(np.linalg.norm(y_adapter))
    return diff / base_norm if base_norm > 0.0 else 0.0
