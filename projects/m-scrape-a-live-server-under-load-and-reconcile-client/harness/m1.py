import ref


def check(workdir):
    from vllm_metrics.scraper import parse_exposition
    from vllm_metrics.histogram import reconstruct_histogram

    raw_text = ref.generate_mock_exposition()
    parsed = parse_exposition(raw_text)

    out = {"buckets_matched": 0.0, "_note": ""}
    if not parsed:
        out["_note"] = "parse_exposition returned empty dictionary"
        return out

    bucket_key = "vllm:request_latency_seconds_bucket"
    if bucket_key not in parsed:
        out["_note"] = f"missing key {bucket_key} in parsed exposition"
        return out

    hist = reconstruct_histogram(parsed[bucket_key])
    if not hist:
        out["_note"] = "reconstruct_histogram returned empty list"
        return out

    les = [h[0] for h in hist]
    if les != sorted(les):
        out["_note"] = "histogram buckets are not sorted by upper bound le"
        return out

    if len(hist) >= 3:
        out["buckets_matched"] = 3.0
    else:
        out["buckets_matched"] = float(len(hist))
    return out
