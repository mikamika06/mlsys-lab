def compute_batch_token_utilization(logs: list) -> dict:
    """Analyze router event logs to compute batch token utilization statistics."""
    if not logs:
        return {
            "mean_prefill_utilization": 0.0,
            "mean_total_utilization": 0.0,
            "peak_prefill_tokens": 0,
            "peak_total_tokens": 0,
        }

    prefill_utils = []
    total_utils = []
    peak_prefill = 0
    peak_total = 0

    for entry in logs:
        prefill_tok = entry.get("prefill_tokens", 0)
        total_tok = entry.get("total_tokens", 0)
        max_prefill = entry.get("max_batch_prefill_tokens", 1)
        max_total = entry.get("max_batch_total_tokens", 1)

        p_util = prefill_tok / float(max_prefill) if max_prefill > 0 else 0.0
        t_util = total_tok / float(max_total) if max_total > 0 else 0.0

        prefill_utils.append(p_util)
        total_utils.append(t_util)

        if prefill_tok > peak_prefill:
            peak_prefill = prefill_tok
        if total_tok > peak_total:
            peak_total = total_tok

    return {
        "mean_prefill_utilization": float(sum(prefill_utils) / len(prefill_utils)),
        "mean_total_utilization": float(sum(total_utils) / len(total_utils)),
        "peak_prefill_tokens": int(peak_prefill),
        "peak_total_tokens": int(peak_total),
    }
