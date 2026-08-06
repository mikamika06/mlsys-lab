import math


def calculate_percentile(data: list[float], p: float, method: str = "nearest") -> float:
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


def evaluate_slo(requests: list[dict], slo_ttft_ms: float, slo_tpot_ms: float, duration_s: float) -> dict:
    total_tokens = sum(r.get("output_tokens", 0) for r in requests)
    good_tokens = sum(
        r.get("output_tokens", 0)
        for r in requests
        if r.get("ttft_ms", float("inf")) <= slo_ttft_ms and r.get("tpot_ms", float("inf")) <= slo_tpot_ms
    )
    throughput = total_tokens / duration_s if duration_s > 0 else 0.0
    goodput = good_tokens / duration_s if duration_s > 0 else 0.0
    goodput_ratio = goodput / throughput if throughput > 0 else 0.0
    return {
        "throughput": throughput,
        "goodput": goodput,
        "goodput_ratio": goodput_ratio,
    }


def rank_configs(configs: list[dict], slo_ttft_ms: float, slo_tpot_ms: float) -> list[dict]:
    evaluated = []
    for cfg in configs:
        cid = cfg["config_id"]
        dur = cfg["duration_s"]
        reqs = cfg["requests"]
        res = evaluate_slo(reqs, slo_ttft_ms, slo_tpot_ms, dur)
        evaluated.append({
            "config_id": cid,
            "goodput": res["goodput"],
            "throughput": res["throughput"],
            "goodput_ratio": res["goodput_ratio"],
        })
    evaluated.sort(key=lambda x: (-x["goodput"], -x["throughput"], str(x["config_id"])))
    for rank, item in enumerate(evaluated, start=1):
        item["rank"] = rank
    return evaluated


def _make_requests(seed: int, count: int) -> list[dict]:
    reqs = []
    for i in range(count):
        val = (seed * 1103515245 + i * 12345 + 12345) & 0x7FFFFFFF
        q = float((val % 150) + 5)
        p = float(((val >> 3) % 200) + 10)
        d = float(((val >> 6) % 500) + 20)
        tokens = int(((val >> 2) % 128) + 16)
        ttft = q + p
        tpot = d / max(1, tokens)
        reqs.append({
            "queue_ms": q,
            "prefill_ms": p,
            "decode_ms": d,
            "output_tokens": tokens,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
        })
    return reqs


SAMPLE_REQUESTS = _make_requests(seed=42, count=200)

CONFIGS = []
for c_idx in range(1, 13):
    reqs = _make_requests(seed=100 + c_idx, count=100 + c_idx * 15)
    CONFIGS.append({
        "config_id": f"cfg_{c_idx}",
        "duration_s": 10.0,
        "requests": reqs,
    })

SLO_TTFT_MS = 120.0
SLO_TPOT_MS = 4.0
