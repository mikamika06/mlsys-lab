import numpy as np

LOG_SAMPLES = [
    """[2026-08-01 10:14:02] ERROR mlx_lm.server: Exception in request handling:
MemoryLimitError: [metal::malloc] Cannot allocate 4096.50 MB, exceeds device limit 16384.00 MB.
Context: batch_size=8, active_tokens=4096. Server shutting down.""",
    """[2026-08-01 11:22:15] ERROR mlx_lm.server: Out of memory during prompt evaluation.
MemoryLimitError: [metal::malloc] Cannot allocate 2048.25 MB, exceeds device limit 8192.00 MB.
Context: batch_size=4, active_tokens=2048. Terminated.""",
]

CONFIG_SAMPLES = [
    {"quantization": {"bits": 4, "group_size": 128}},
    {"bits": 8, "group_size": 32},
    {"model_type": "llama"},
]


def parse_oom_log(log_text):
    import re
    req = float(re.search(r"Cannot allocate\s+([0-9.]+)\s*MB", log_text).group(1))
    lim = float(re.search(r"limit\s+([0-9.]+)\s*MB", log_text).group(1))
    tok = int(re.search(r"active_tokens\s*=\s*(\d+)", log_text).group(1))
    bs = int(re.search(r"batch_size\s*=\s*(\d+)", log_text).group(1))
    return {
        "requested_mb": req,
        "limit_mb": lim,
        "peak_mb": req + lim,
        "active_tokens": tok,
        "batch_size": bs,
        "is_oom": True,
    }


def extract_quant_config(config_dict):
    q = config_dict.get("quantization", {})
    bits = q.get("bits", config_dict.get("bits", 16))
    gs = q.get("group_size", config_dict.get("group_size", 64))
    return {"bits": int(bits), "group_size": int(gs), "is_quantized": int(bits) < 16}


def simulate_quant_dequant(weights, bits=4, group_size=64):
    weights = np.array(weights, dtype=np.float32)
    orig_shape = weights.shape
    flat = weights.flatten()
    pad_len = (group_size - (flat.size % group_size)) % group_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant")
    reshaped = flat.reshape(-1, group_size)
    max_val = np.maximum(np.max(np.abs(reshaped), axis=1, keepdims=True), 1e-8)
    qmax = (1 << (bits - 1)) - 1
    scales = max_val / qmax
    q = np.clip(np.round(reshaped / scales), -qmax, qmax)
    dq = (q * scales).reshape(-1)[: weights.size]
    return dq.reshape(orig_shape)


def evaluate_weight_drift(weights, bits=4, group_size=64, max_allowed_mse=0.05):
    dq = simulate_quant_dequant(weights, bits=bits, group_size=group_size)
    mse = float(np.mean((weights - dq) ** 2))
    return {"mse": mse, "exceeds_threshold": mse > max_allowed_mse, "max_allowed_mse": float(max_allowed_mse)}
