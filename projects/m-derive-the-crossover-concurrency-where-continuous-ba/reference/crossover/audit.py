def audit_benchmark(b):
    issues = []
    if b.get("precision") not in ("FP16",):
        issues.append("quantization_mismatch")
    if not b.get("warmup", True):
        issues.append("no_warmup")
    if b.get("requests", 10) <= 1:
        issues.append("single_request")
    return issues
