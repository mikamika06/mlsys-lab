import ref


def check(workdir):
    from vllm_metrics.scraper import parse_exposition
    from vllm_metrics.histogram import reconstruct_histogram
    from vllm_metrics.reconcile import interpolate_percentile, reconcile_latency

    raw_text = ref.generate_mock_exposition()
    parsed = parse_exposition(raw_text)
    hist = reconstruct_histogram(parsed["vllm:request_latency_seconds_bucket"])
    client_latencies = ref.generate_client_latencies()

    res = reconcile_latency(client_latencies, hist)

    out = {"rel_err_match": 0.0, "reconciliation_match": 0.0, "_note": ""}
    if not isinstance(res, dict) or "rel_err" not in res:
        out["_note"] = "reconcile_latency must return a dict containing 'rel_err'"
        return out

    rel_err = res["rel_err"]
    if 0.0 <= rel_err <= 5.0:
        out["rel_err_match"] = 1.0
    else:
        out["_note"] = f"rel_err {rel_err} out of reasonable bounds"

    p99 = interpolate_percentile(hist, 0.99)
    if p99 > 0:
        out["reconciliation_match"] = 1.0
    else:
        out["_note"] = "interpolated percentile returned non-positive value"

    return out
