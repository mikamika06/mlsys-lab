def classify_error(log_text):
    """Classify converter log text into a root cause category."""
    text = (log_text or "").lower()
    if "opcode not found" in text or "unsupported operator" in text or "op not supported" in text:
        return "unsupported_op"
    if "shape mismatch" in text or "dimension mismatch" in text or "incompatible shape" in text:
        return "shape_mismatch"
    if "quantization" in text or "calibration failed" in text or "scale factor" in text:
        return "quantization_error"
    if "allocation overflow" in text or "out of memory" in text or "buffer overflow" in text:
        return "memory_limit"
    return "unknown"
