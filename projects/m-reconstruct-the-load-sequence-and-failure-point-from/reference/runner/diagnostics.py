def classify_missing_asset(error_context):
    msg = error_context.get("message", "").lower()
    if "manifest" in msg or "config.json" in msg or "index.json" in msg:
        return "manifest_not_found"
    if "model" in msg or "weights" in msg or "safetensors" in msg or "bin" in msg:
        return "model_not_found"
    return "unknown_asset_error"


def check_cpu_fallback(session_metrics):
    device = session_metrics.get("device", "cpu")
    kernels = session_metrics.get("cuda_kernels_executed", 0)
    cpu_ops = session_metrics.get("cpu_fallback_ops", 0)
    if device == "cuda" and cpu_ops > 0 and kernels == 0:
        return True
    if device == "cuda" and cpu_ops > (kernels * 0.5):
        return True
    return False
