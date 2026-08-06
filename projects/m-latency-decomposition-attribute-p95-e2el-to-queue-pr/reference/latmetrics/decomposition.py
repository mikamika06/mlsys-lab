import math


def calculate_percentile(data: list[float], p: float, method: str = "nearest") -> float:
    """Calculate p-th percentile using nearest-rank or linear interpolation."""
    if not data:
        raise ValueError("data cannot be empty")
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return float(sorted_data[0])
    if method == "nearest":
        rank = math.ceil((p / 100.0) * n)
        idx = max(0, min(n - 1, rank - 1))
        return float(sorted_data[idx])
    elif method == "linear":
        pos = (p / 100.0) * (n - 1)
        lower = int(pos)
        upper = min(lower + 1, n - 1)
        weight = pos - lower
        return float(sorted_data[lower] + weight * (sorted_data[upper] - sorted_data[lower]))
    else:
        raise ValueError(f"Unknown method: {method}")


def decompose_latencies(requests: list[dict], method: str = "nearest") -> dict:
    """Decompose p95 E2E latency into queue, prefill, and decode components."""
    if not requests:
        return {}
    queue_lats = [r["queue_ms"] for r in requests]
    prefill_lats = [r["prefill_ms"] for r in requests]
    decode_lats = [r["decode_ms"] for r in requests]
    e2e_lats = [r["queue_ms"] + r["prefill_ms"] + r["decode_ms"] for r in requests]

    p95_queue = calculate_percentile(queue_lats, 95.0, method=method)
    p95_prefill = calculate_percentile(prefill_lats, 95.0, method=method)
    p95_decode = calculate_percentile(decode_lats, 95.0, method=method)
    p95_e2e = calculate_percentile(e2e_lats, 95.0, method=method)

    comp_sum = p95_queue + p95_prefill + p95_decode
    if comp_sum > 0:
        queue_share = p95_queue / comp_sum
        prefill_share = p95_prefill / comp_sum
        decode_share = p95_decode / comp_sum
    else:
        queue_share = prefill_share = decode_share = 0.0

    return {
        "p95_e2e": p95_e2e,
        "p95_queue": p95_queue,
        "p95_prefill": p95_prefill,
        "p95_decode": p95_decode,
        "queue_share": queue_share,
        "prefill_share": prefill_share,
        "decode_share": decode_share,
    }
