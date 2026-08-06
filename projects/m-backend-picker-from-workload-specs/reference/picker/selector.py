def select_backend(spec: dict) -> str:
    device = spec.get("device", "gpu")
    if device == "cpu":
        return "ort_cpu"

    model_type = spec.get("model_type", "standard")
    has_custom_ops = spec.get("has_custom_ops", False)
    seq_len = spec.get("seq_len", 1)
    total_inferences = spec.get("total_inferences", 1000)
    target_latency_ms = spec.get("target_latency_ms", 10.0)

    if model_type == "llm":
        if seq_len >= 128 and not has_custom_ops:
            return "trt_llm"
        return "ort_cuda"

    if has_custom_ops:
        return "ort_cuda"

    if total_inferences < 500:
        return "ort_cuda"

    if target_latency_ms < 3.0:
        return "standalone_trt"

    return "ort_trt"
