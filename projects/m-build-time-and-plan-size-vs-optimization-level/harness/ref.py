import hashlib

SAMPLE_CONFIGS = [
    {"optimization_level": 0, "build_time_sec": 1.2, "plan_size_bytes": 10485760},
    {"optimization_level": 1, "build_time_sec": 3.5, "plan_size_bytes": 8912896},
    {"optimization_level": 2, "build_time_sec": 8.1, "plan_size_bytes": 7864320},
    {"optimization_level": 3, "build_time_sec": 18.4, "plan_size_bytes": 7340032},
    {"optimization_level": 4, "build_time_sec": 42.0, "plan_size_bytes": 7130316},
    {"optimization_level": 5, "build_time_sec": 95.2, "plan_size_bytes": 7077888},
]

FAILURE_LOGS = [
    ("ONNX Parser: failed to parse node /conv1/Conv", "parser"),
    ("Network error: input tensor dimension mismatch at node 4", "network"),
    ("BuilderConfig error: requested workspace 4096MB exceeds limit", "builder_config"),
    ("Engine build failed: serialization write error on output plan", "engine"),
    ("Unknown fault occurred", "unknown"),
]


def reference_analyze_build_tradeoffs(configs):
    results = []
    base_size = None
    for cfg in configs:
        if cfg["optimization_level"] == 0:
            base_size = float(cfg["plan_size_bytes"])
            break
    if base_size is None or base_size == 0:
        base_size = float(configs[0]["plan_size_bytes"]) if configs else 1.0

    for cfg in configs:
        level = cfg["optimization_level"]
        build_time = cfg["build_time_sec"]
        plan_size = cfg["plan_size_bytes"]
        ratio = float(plan_size) / base_size
        results.append({
            "optimization_level": level,
            "build_time_sec": build_time,
            "plan_size_bytes": plan_size,
            "size_ratio": ratio,
        })
    return results


def reference_classify_failure(exception_log):
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


def reference_verify_roundtrip(engine_plan):
    if not isinstance(engine_plan, (bytes, bytearray)):
        return False, "invalid_plan_type"
    
    header = engine_plan[:8]
    if len(header) < 8 or not header.startswith(b"TRT"):
        return False, "corrupted_header"
    
    body = engine_plan[8:]
    reconstructed = header + body
    if reconstructed != engine_plan:
        return False, "mismatch"
    
    digest = hashlib.sha256(engine_plan).hexdigest()
    return True, digest
