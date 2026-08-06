from capacity.cost import compute_cost_per_million_tokens

def select_cheapest_config(candidates: list[dict], p99_slo_ms: float) -> dict:
    """Select cheapest configuration satisfying the p99 latency SLO."""
    valid_candidates = []
    for cand in candidates:
        if cand["p99_latency_ms"] <= p99_slo_ms:
            cost = compute_cost_per_million_tokens(
                cand["measured_throughput_tok_per_sec"],
                cand["hourly_instance_price"]
            )
            item = dict(cand)
            item["cost_per_m_tokens"] = cost
            valid_candidates.append(item)

    if not valid_candidates:
        raise ValueError("No configuration meets the required p99 SLO")

    valid_candidates.sort(key=lambda x: (x["cost_per_m_tokens"], x["p99_latency_ms"]))
    return valid_candidates[0]
