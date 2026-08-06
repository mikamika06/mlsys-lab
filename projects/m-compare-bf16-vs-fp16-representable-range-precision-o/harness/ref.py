import numpy as np

def generate_extreme_tensors():
    np.random.seed(42)
    tensors = [
        np.array([1.0, 2.0, 1000.0, 70000.0, 1e38], dtype=np.float32),
        np.random.uniform(-50.0, 50.0, size=(8, 8)).astype(np.float32),
        np.array([0.0, -1.0, 65504.0, 3.3895314e38], dtype=np.float32)
    ]
    return tensors

def analyze_ranges(tensor):
    arr = np.asarray(tensor, dtype=np.float32)
    u = arr.view(np.uint32)
    bf16_bits = u & 0xFFFF0000
    bf16_val = bf16_bits.view(np.float32)
    fp16_overflow = np.abs(arr) > 65504.0
    bf16_overflow = np.abs(arr) > 3.3895314e38
    return {
        "bf16_approx": bf16_val,
        "fp16_overflow": fp16_overflow,
        "bf16_overflow": bf16_overflow
    }

def quantify_roundtrip_loss(tensor):
    arr = np.asarray(tensor, dtype=np.float32)
    u = arr.view(np.uint32)
    bf16_u = u & 0xFFFF0000
    roundtrip = bf16_u.view(np.float32)
    diff = np.abs(arr - roundtrip)
    rel_err = np.max(diff / (np.abs(arr) + 1e-12))
    return float(rel_err)
