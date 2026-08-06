def evaluate_slo(requests: list[dict], slo_ttft_ms: float, slo_tpot_ms: float, duration_s: float) -> dict:
    """Evaluate SLO compliance and compute throughput vs goodput."""
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
    """Rank serving configurations by goodput."""
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
