def compute_latency_ratios(agg_results: list[dict], disagg_results: list[dict]) -> dict:
    """
    Computes summary metrics and latency ratios between disaggregated and aggregated runs.
    Returns dict containing:
      - agg_mean_ttft
      - disagg_mean_ttft
      - agg_mean_itl
      - disagg_mean_itl
      - ttft_ratio (disagg / agg)
      - itl_ratio (disagg / agg)
      - latency_ratio (disagg_total_mean_latency / agg_total_mean_latency)
    """
    raise NotImplementedError
