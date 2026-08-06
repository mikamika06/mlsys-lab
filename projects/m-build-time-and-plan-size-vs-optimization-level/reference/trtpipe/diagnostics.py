def classify_failure(exception_log):
    """Classify the failure stage of a TensorRT build pipeline based on log output."""
    log_lower = exception_log.lower()
    if "parser" in log_lower or "onnx" in log_lower or "import" in log_lower:
        return "parser"
    if "network" in log_lower or "layer" in log_lower or "tensorspec" in log_lower:
        return "network"
    if "builderconfig" in log_lower or "config" in log_lower or "tactic" in log_lower or "workspace" in log_lower:
        return "builder_config"
    if "engine" in log_lower or "serialize" in log_lower or "plan" in log_lower:
        return "engine"
    return "unknown"
