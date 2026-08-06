def classify_log_symptom(log_snippet: str) -> str:
    """Classify a training log symptom to its root numerical cause."""
    text = log_snippet.lower()
    if "loss scale" in text or "scale to 1" in text or "fp16" in text:
        return "fp16_overflow"
    elif "bfloat16" in text or "bf16" in text or "exact 0" in text or "underflow" in text:
        return "bf16_underflow"
    elif "activation" in text or "exceeded max_val" in text or "dynamic range" in text:
        return "activation_overflow"
    elif "checkpoint" in text or "nf4" in text or "quant" in text or "drift" in text:
        return "quantization_drift"
    return "unknown"
