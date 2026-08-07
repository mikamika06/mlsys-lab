import numpy as np

def oracle_compute_bpw_and_size(params: int, bits: float) -> tuple:
    size_bytes = int(params * bits / 8.0)
    return float(bits), size_bytes

def oracle_measure_peak_memory(params: int, bits: float, overhead_mb: float) -> float:
    model_mem_mb = (params * bits / 8.0) / (1024 * 1024)
    return float(model_mem_mb + overhead_mb)

def oracle_measure_quality(logits_ref: np.ndarray, logits_quant: np.ndarray) -> dict:
    eps = 1e-10
    p = np.exp(logits_ref - np.max(logits_ref, axis=-1, keepdims=True))
    p /= np.sum(p, axis=-1, keepdims=True)
    q = np.exp(logits_quant - np.max(logits_quant, axis=-1, keepdims=True))
    q /= np.sum(q, axis=-1, keepdims=True)
    kld = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1).mean()
    ce = -np.sum(p * np.log(q + eps), axis=-1).mean()
    ppl = float(np.exp(ce))
    return {"kld": float(kld), "ppl": ppl}

def oracle_measure_speed(bpw: float, base_tps: float) -> float:
    factor = 1.0 + (16.0 - bpw) * 0.03
    return float(base_tps * factor)

def oracle_generate_recommendation_table(configs: list, memory_limits: list) -> list:
    table = []
    for limit in memory_limits:
        best = None
        for cfg in configs:
            if cfg["peak_memory_mb"] <= limit * 1024:
                if best is None or cfg["bpw"] > best["bpw"]:
                    best = cfg
        table.append({"memory_limit_gb": limit, "recommended_recipe": best["name"] if best else "none"})
    return table

def oracle_auto_select_recipe(available_ram_gb: float, table: list) -> str:
    selected = "none"
    for row in table:
        if available_ram_gb >= row["memory_limit_gb"]:
            selected = row["recommended_recipe"]
    return selected
