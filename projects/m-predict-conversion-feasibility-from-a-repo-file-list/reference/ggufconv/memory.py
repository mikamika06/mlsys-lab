import math


def _dtype_bytes(dt: str) -> int:
    if dt in ("float32", "f32"):
        return 4
    if dt in ("float16", "f16", "bfloat16", "bf16"):
        return 2
    if dt in ("int8", "i8", "q8_0"):
        return 1
    return 4


def estimate_conversion_memory(
    tensors: list[dict], lazy: bool = True, base_overhead_mb: float = 256.0
) -> dict:
    base_bytes = int(base_overhead_mb * 1024 * 1024)
    if not tensors:
        return {
            "peak_memory_bytes": base_bytes,
            "lazy": bool(lazy),
            "total_model_bytes": 0,
        }

    sizes = []
    shard_map = {}
    for t in tensors:
        sz = math.prod(t["shape"]) * _dtype_bytes(t.get("dtype", "float32"))
        sizes.append(sz)
        sid = t.get("shard_id", "default")
        if sid is None:
            sid = "default"
        shard_map.setdefault(sid, []).append(sz)

    total_bytes = sum(sizes)

    if not lazy:
        max_t = max(sizes)
        peak = base_bytes + total_bytes + max_t
    else:
        shard_peaks = []
        for sid, s_sizes in shard_map.items():
            shard_peaks.append(sum(s_sizes) + max(s_sizes))
        peak = base_bytes + max(shard_peaks)

    return {
        "peak_memory_bytes": int(peak),
        "lazy": bool(lazy),
        "total_model_bytes": int(total_bytes),
    }
