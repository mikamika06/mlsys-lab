import numpy as np

def create_tensor(shape, dtype="float16", aligned=True):
    return {
        "shape": shape,
        "dtype": dtype,
        "aligned": aligned,
        "contiguous": True
    }

def get_backend(q, k, v, mask=None):
    if q["dtype"] not in ["float16", "bfloat16"] or not q["aligned"]:
        return "math"
    if mask is not None and not mask.get("supported", True):
        return "math"
    return "flash_attention"

def find_disqualification_reason(q, k, v, mask=None):
    if q["dtype"] not in ["float16", "bfloat16"]:
        return "unsupported_dtype"
    if not q["aligned"]:
        return "bad_alignment"
    if mask is not None and not mask.get("supported", True):
        return "unsupported_mask"
    return "none"

def fix_inputs(q, k, v, mask=None):
    q_fixed = q.copy()
    q_fixed["dtype"] = "float16"
    q_fixed["aligned"] = True
    k_fixed = k.copy()
    k_fixed["dtype"] = "float16"
    k_fixed["aligned"] = True
    v_fixed = v.copy()
    v_fixed["dtype"] = "float16"
    v_fixed["aligned"] = True
    m_fixed = None
    if mask is not None:
        m_fixed = mask.copy()
        m_fixed["supported"] = True
    return q_fixed, k_fixed, v_fixed, m_fixed

def measure_speedup(q, k, v, mask=None):
    b_before = get_backend(q, k, v, mask)
    q_f, k_f, v_f, m_f = fix_inputs(q, k, v, mask)
    b_after = get_backend(q_f, k_f, v_f, m_f)
    t_before = 5.0 if b_before == "math" else 1.0
    t_after = 1.0 if b_after == "flash_attention" else 5.0
    return t_before / t_after

def run_configs(configs):
    results = []
    for cfg in configs:
        b = get_backend(cfg["q"], cfg["k"], cfg["v"], cfg.get("mask"))
        results.append({"config": cfg, "backend": b})
    return results

def strict_attention(q, k, v, mask=None, strict=True):
    b = get_backend(q, k, v, mask)
    if strict and b == "math":
        raise RuntimeError("Silent fallback to math backend detected!")
    return f"attended_with_{b}"
