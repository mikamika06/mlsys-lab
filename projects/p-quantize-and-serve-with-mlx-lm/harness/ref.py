import numpy as np

def simulate_quantization(bits: int, target_gb: float) -> dict:
    weights_size = 32.0 * (bits / 16.0)
    valid = weights_size <= target_gb
    return {"size_gb": weights_size, "valid": valid, "quantize_ok": 1.0 if valid else 0.0}

def compute_perplexity_diff(logits_orig, logits_quant) -> float:
    diff = np.abs(logits_orig - logits_quant)
    return float(np.mean(diff))

def apply_chat_template(messages: list) -> str:
    formatted = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted

def run_load_test(concurrency: int, latencies: list) -> dict:
    p95 = float(np.percentile(latencies, 95))
    ok = (concurrency >= 3) and (p95 < 2.0)
    return {"p95": p95, "load_p95_ok": 1.0 if ok else 0.0}

def check_memory_stability(allocations: list) -> dict:
    diffs = np.diff(allocations)
    max_growth = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    stable = max_growth < 0.01
    return {"max_growth": max_growth, "memory_stable_ok": 1.0 if stable else 0.0}

def verify_deployment_script(script_text: str) -> bool:
    return "mlx_lm" in script_text and "serve" in script_text
