import random

def is_eligible(backend, dtype, is_causal, q_len, kv_len, head_dim, device_cap):
    if backend == "flash_attention":
        if device_cap < (8, 0):
            return False
        if dtype not in ("float16", "bfloat16"):
            return False
        if head_dim > 128 or head_dim % 8 != 0:
            return False
        return True
    elif backend == "mem_efficient":
        if device_cap < (7, 0):
            return False
        if dtype not in ("float16", "bfloat16", "float32"):
            return False
        if head_dim > 256:
            return False
        return True
    elif backend == "math":
        return True
    return False

def predict_backend(dtype, is_causal, q_len, kv_len, head_dim, device_cap):
    for b in ["flash_attention", "mem_efficient", "math"]:
        if is_eligible(b, dtype, is_causal, q_len, kv_len, head_dim, device_cap):
            return b
    return "math"

def detect_backend_from_trace(events):
    for ev in events:
        name = ev.get("name", "").lower()
        if "flash_attn" in name or "flashattention" in name:
            return "flash_attention"
        if "mem_efficient" in name or "efficient_attention" in name:
            return "mem_efficient"
        if "math" in name or "sdpa_kernel_math" in name:
            return "math"
    return "math"

CONFIGS = []
rng = random.Random(42)
dtypes = ["float16", "bfloat16", "float32"]
caps = [(7, 0), (8, 0), (9, 0)]
for _ in range(30):
    dt = rng.choice(dtypes)
    causal = rng.choice([True, False])
    ql = rng.choice([128, 512, 1024, 2048])
    kvl = rng.choice([128, 512, 1024, 2048])
    hd = rng.choice([32, 64, 128, 256])
    cap = rng.choice(caps)
    CONFIGS.append({
        "dtype": dt,
        "is_causal": causal,
        "q_len": ql,
        "kv_len": kvl,
        "head_dim": hd,
        "device_cap": cap
    })
