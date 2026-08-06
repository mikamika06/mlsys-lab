def compute_request_goodput(req: dict, penalty_factor: float = 0.5) -> dict:
    total_sec = req["duration_ms"] / 1000.0
    if total_sec <= 0:
        return {"goodput_tps": 0.0, "meets_sla": False}

    meets_sla = req["latency_ms"] <= req["sla_latency_ms"]
    if not meets_sla:
        return {"goodput_tps": 0.0, "meets_sla": False}

    accepted = req.get("accepted_tokens", req.get("generated_tokens", 0))
    rejected = req.get("rejected_tokens", 0)

    effective_tokens = max(0.0, float(accepted) - penalty_factor * float(rejected))
    goodput_tps = effective_tokens / total_sec
    return {"goodput_tps": goodput_tps, "meets_sla": True}


def aggregate_goodput_comparison(scheduler_log: list[dict], penalty_factor: float = 0.5) -> dict:
    spec_tokens = 0.0
    spec_time = 0.0
    base_tokens = 0.0
    base_time = 0.0

    spec_goodput_sum = 0.0
    base_goodput_sum = 0.0
    spec_count = 0
    base_count = 0

    for entry in scheduler_log:
        res = compute_request_goodput(entry, penalty_factor=penalty_factor)
        tps = res["goodput_tps"]
        dur = entry["duration_ms"] / 1000.0

        if entry.get("is_speculative", False):
            spec_count += 1
            spec_goodput_sum += tps
            if res["meets_sla"]:
                accepted = entry.get("accepted_tokens", 0)
                rejected = entry.get("rejected_tokens", 0)
                spec_tokens += max(0.0, float(accepted) - penalty_factor * float(rejected))
            spec_time += dur
        else:
            base_count += 1
            base_goodput_sum += tps
            if res["meets_sla"]:
                base_tokens += entry.get("generated_tokens", 0)
            base_time += dur

    agg_spec = spec_tokens / spec_time if spec_time > 0 else 0.0
    agg_base = base_tokens / base_time if base_time > 0 else 0.0
    ratio = agg_spec / agg_base if agg_base > 0 else 0.0

    return {
        "spec_aggregate_goodput_tps": agg_spec,
        "base_aggregate_goodput_tps": agg_base,
        "throughput_ratio": ratio,
        "spec_count": spec_count,
        "base_count": base_count,
    }
