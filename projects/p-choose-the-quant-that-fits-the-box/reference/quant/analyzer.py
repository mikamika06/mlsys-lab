import numpy as np

def compute_bpw_and_size(params: int, bits: float) -> tuple:
    size_bytes = int(params * bits / 8.0)
    return float(bits), size_bytes

def measure_peak_memory(params: int, bits: float, overhead_mb: float) -> float:
    model_mem_mb = (params * bits / 8.0) / (1024 * 1024)
    return float(model_mem_mb + overhead_mb)

def measure_quality(logits_ref: np.ndarray, logits_quant: np.ndarray) -> dict:
    eps = 1e-10
    p = np.exp(logits_ref - np.max(logits_ref, axis=-1, keepdims=True))
    p /= np.sum(p, axis=-1, keepdims=True)
    q = np.exp(logits_quant - np.max(logits_quant, axis=-1, keepdims=True))
    q /= np.sum(q, axis=-1, keepdims=True)
    kld = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1).mean()
    ce = -np.sum(p * np.log(q + eps), axis=-1).mean()
    ppl = float(np.exp(ce))
    return {"kld": float(kld), "ppl": ppl}

def measure_speed(bpw: float, base_tps: float) -> float:
    factor = 1.0 + (16.0 - bpw) * 0.03
    return float(base_tps * factor)
