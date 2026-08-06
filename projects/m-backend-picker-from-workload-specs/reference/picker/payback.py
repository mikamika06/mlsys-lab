import math


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
