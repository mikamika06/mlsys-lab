import math

WORKLOADS = [
    {"device": "cpu", "model_type": "standard", "has_custom_ops": False, "seq_len": 1, "total_inferences": 100, "target_latency_ms": 15.0},
    {"device": "gpu", "model_type": "llm", "has_custom_ops": False, "seq_len": 512, "total_inferences": 10000, "target_latency_ms": 20.0},
    {"device": "gpu", "model_type": "llm", "has_custom_ops": True, "seq_len": 256, "total_inferences": 5000, "target_latency_ms": 10.0},
    {"device": "gpu", "model_type": "standard", "has_custom_ops": True, "seq_len": 1, "total_inferences": 10000, "target_latency_ms": 1.0},
    {"device": "gpu", "model_type": "standard", "has_custom_ops": False, "seq_len": 1, "total_inferences": 200, "target_latency_ms": 1.0},
    {"device": "gpu", "model_type": "standard", "has_custom_ops": False, "seq_len": 1, "total_inferences": 50000, "target_latency_ms": 1.5},
    {"device": "gpu", "model_type": "standard", "has_custom_ops": False, "seq_len": 1, "total_inferences": 50000, "target_latency_ms": 8.0},
]

CANDIDATES = [
    {"backend": "ort_cuda", "latency_ms": 10.0, "build_time_sec": 0.0},
    {"backend": "ort_trt", "latency_ms": 5.0, "build_time_sec": 30.0},
    {"backend": "standalone_trt", "latency_ms": 2.5, "build_time_sec": 120.0},
    {"backend": "ort_cpu", "latency_ms": 50.0, "build_time_sec": 0.0},
]


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


def calculate_payback_volume(build_time_sec: float, base_latency_ms: float, target_latency_ms: float) -> int:
    if target_latency_ms >= base_latency_ms:
        return -1
    delta_lat_sec = (base_latency_ms - target_latency_ms) / 1000.0
    if delta_lat_sec <= 0:
        return -1
    return math.ceil(build_time_sec / delta_lat_sec)


def build_normalized_table(candidates: list[dict], baseline_backend: str = "ort_cuda") -> list[dict]:
    base_item = next((c for c in candidates if c["backend"] == baseline_backend), None)
    if base_item is None and candidates:
        base_item = candidates[0]

    base_lat = base_item["latency_ms"] if base_item else 1.0

    out = []
    for c in candidates:
        lat = c["latency_ms"]
        b_time = c.get("build_time_sec", 0.0)
        speedup = base_lat / lat if lat > 0 else 0.0
        payback = calculate_payback_volume(b_time, base_lat, lat)

        out.append({
            "backend": c["backend"],
            "speedup": round(speedup, 4),
            "build_penalty_sec": round(b_time, 2),
            "payback_volume": payback,
            "is_viable": lat < base_lat or c["backend"] == baseline_backend
        })
    return out
