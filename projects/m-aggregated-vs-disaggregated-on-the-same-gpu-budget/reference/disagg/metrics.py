def compute_latency_ratios(agg_results: list[dict], disagg_results: list[dict]) -> dict:
    """Computes summary metrics and ratios between disaggregated and aggregated runs."""
    if not agg_results or not disagg_results:
        return {}

    agg_ttft = [r["ttft"] for r in agg_results]
    disagg_ttft = [r["ttft"] for r in disagg_results]

    agg_itl = [r["avg_itl"] for r in agg_results]
    disagg_itl = [r["avg_itl"] for r in disagg_results]

    agg_lat = [r["total_latency"] for r in agg_results]
    disagg_lat = [r["total_latency"] for r in disagg_results]

    agg_mean_ttft = sum(agg_ttft) / len(agg_ttft)
    disagg_mean_ttft = sum(disagg_ttft) / len(disagg_ttft)

    agg_mean_itl = sum(agg_itl) / len(agg_itl)
    disagg_mean_itl = sum(disagg_itl) / len(disagg_itl)

    agg_mean_lat = sum(agg_lat) / len(agg_lat)
    disagg_mean_lat = sum(disagg_lat) / len(disagg_lat)

    return {
        "agg_mean_ttft": agg_mean_ttft,
        "disagg_mean_ttft": disagg_mean_ttft,
        "agg_mean_itl": agg_mean_itl,
        "disagg_mean_itl": disagg_mean_itl,
        "ttft_ratio": disagg_mean_ttft / agg_mean_ttft if agg_mean_ttft > 0 else 1.0,
        "itl_ratio": disagg_mean_itl / agg_mean_itl if agg_mean_itl > 0 else 1.0,
        "latency_ratio": disagg_mean_lat / agg_mean_lat if agg_mean_lat > 0 else 1.0,
    }
