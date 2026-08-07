def diff_responses(native_resp, shim_resp):
    """Diff native and shim response structures."""
    missing_in_shim = []
    extra_in_shim = []
    type_mismatches = []

    native_headers = set(native_resp.get("headers", {}).keys())
    shim_headers = set(shim_resp.get("headers", {}).keys())

    missing_headers = sorted(list(native_headers - shim_headers))
    extra_headers = sorted(list(shim_headers - native_headers))

    native_chunks = native_resp.get("chunks", [])
    shim_chunks = shim_resp.get("chunks", [])

    n_len = len(native_chunks)
    s_len = len(shim_chunks)
    chunk_count_diff = s_len - n_len

    def extract_keys(obj, prefix=""):
        keys = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else k
                keys[p] = type(v).__name__
                keys.update(extract_keys(v, p))
        elif isinstance(obj, list) and obj:
            p = f"{prefix}[]"
            keys[p] = "list"
            keys.update(extract_keys(obj[0], p))
        return keys

    min_len = min(n_len, s_len)
    for i in range(min_len):
        n_k = extract_keys(native_chunks[i])
        s_k = extract_keys(shim_chunks[i])

        for k in n_k:
            if k not in s_k:
                if k not in missing_in_shim:
                    missing_in_shim.append(k)
            elif n_k[k] != s_k[k]:
                item = f"{k}:{n_k[k]}!=:{s_k[k]}"
                if item not in type_mismatches:
                    type_mismatches.append(item)

        for k in s_k:
            if k not in n_k and k not in extra_in_shim:
                extra_in_shim.append(k)

    return {
        "missing_in_shim": sorted(missing_in_shim),
        "extra_in_shim": sorted(extra_in_shim),
        "type_mismatches": sorted(type_mismatches),
        "missing_headers": missing_headers,
        "extra_headers": extra_headers,
        "chunk_count_diff": chunk_count_diff,
    }


def recover_timings(event_stream):
    """Recover per-phase timings from event stream timestamps."""
    if not event_stream:
        return {"ttft": 0.0, "inter_token_latencies": [], "mean_tps": 0.0}

    req_time = event_stream[0]["timestamp"]
    first_token_time = None
    token_times = []

    for evt in event_stream:
        if evt.get("event") == "token" or "token" in evt:
            if first_token_time is None:
                first_token_time = evt["timestamp"]
            token_times.append(evt["timestamp"])

    if first_token_time is None:
        return {"ttft": 0.0, "inter_token_latencies": [], "mean_tps": 0.0}

    ttft = round(first_token_time - req_time, 6)
    latencies = [
        round(token_times[i] - token_times[i - 1], 6)
        for i in range(1, len(token_times))
    ]

    total_gen_time = token_times[-1] - first_token_time if len(token_times) > 1 else 0.0
    mean_tps = (
        round((len(token_times) - 1) / total_gen_time, 4)
        if total_gen_time > 0
        else 0.0
    )

    return {
        "ttft": ttft,
        "inter_token_latencies": latencies,
        "mean_tps": mean_tps,
    }


def find_ignored_parameter(runner_func, base_params, param_candidates):
    """Find parameter silently ignored by shim."""
    baseline = runner_func(base_params)

    for param_name, test_values in param_candidates.items():
        is_ignored = True
        for val in test_values:
            params = dict(base_params)
            params[param_name] = val
            res = runner_func(params)
            if res != baseline:
                is_ignored = False
                break
        if is_ignored:
            return param_name

    return None
