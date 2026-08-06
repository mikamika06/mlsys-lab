import ref


def check(workdir):
    out = {"logs_parsed": 0.0, "oom_metrics_matched": 0.0}
    try:
        from mlxdiag.oom import parse_oom_log
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    parsed_count = 0
    matched_count = 0
    for sample in ref.LOG_SAMPLES:
        want = ref.parse_oom_log(sample)
        try:
            got = parse_oom_log(sample)
            parsed_count += 1
            if (
                abs(got.get("requested_mb", 0) - want["requested_mb"]) < 1e-2
                and abs(got.get("limit_mb", 0) - want["limit_mb"]) < 1e-2
                and abs(got.get("peak_mb", 0) - want["peak_mb"]) < 1e-2
                and got.get("active_tokens") == want["active_tokens"]
                and got.get("batch_size") == want["batch_size"]
                and got.get("is_oom") is True
            ):
                matched_count += 1
        except Exception as e:
            out["_note"] = f"parse_oom_log failed: {e}"
            return out

    if parsed_count == len(ref.LOG_SAMPLES):
        out["logs_parsed"] = 1.0
    if matched_count == len(ref.LOG_SAMPLES):
        out["oom_metrics_matched"] = 1.0

    return out
