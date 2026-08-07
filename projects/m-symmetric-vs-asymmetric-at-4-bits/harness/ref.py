import re

CONFIGS = [
    {"bits": 4, "symmetric": True, "group_size": 128, "weights": [0.1, -0.2, 0.3, -0.4]},
    {"bits": 4, "symmetric": False, "group_size": 128, "weights": [0.1, -0.2, 0.3, -0.4]},
    {"bits": 4, "symmetric": True, "group_size": 64, "weights": [-1.0, 0.5, 0.2, -0.3]}
]

BUDGET_CASES = [
    {"total_params": 4096, "target_bits": 4.5, "symmetric": True},
    {"total_params": 8192, "target_bits": 4.0, "symmetric": False},
    {"total_params": 2048, "target_bits": 5.0, "symmetric": True}
]

REGEX_CASES = [
    {"pattern": ".*layer.*", "text": "model.layers.0.self_attn.q_proj"},
    {"pattern": "model\\.layers\\.\\d+\\.mlp", "text": "model.layers.5.mlp.gate_proj"},
    {"pattern": "^mlp$", "text": "mlp"}
]

def simulate_quantize(cfg):
    sym = cfg["symmetric"]
    w = cfg["weights"]
    mx = max(abs(x) for x in w)
    if sym:
        scale = mx / 7.0 if mx > 0 else 1.0
        q = [max(-8, min(7, round(val / scale))) for val in w]
        deq = [val * scale for val in q]
    else:
        mn = min(w)
        mx = max(w)
        rng = mx - mn if mx != mn else 1.0
        scale = rng / 15.0
        zp = round(-mn / scale)
        zp = max(0, min(15, zp))
        q = [max(0, min(15, round(val / scale) + zp)) for val in w]
        deq = [(val - zp) * scale for val in q]
    return {"quantized": q, "dequantized": deq, "symmetric": sym}

def compute_group_size(case):
    bits = case["target_bits"]
    p = case["total_params"]
    gs = 128 if bits <= 4.0 else 64
    if case["symmetric"]:
        gs = min(gs, p)
    else:
        gs = max(32, gs // 2)
    return gs

def match_regex(pattern, text):
    try:
        return bool(re.search(pattern, text))
    except Exception:
        return False
