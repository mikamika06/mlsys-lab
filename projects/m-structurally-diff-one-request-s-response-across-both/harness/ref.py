import copy


NATIVE_RESPONSES = [
    {
        "headers": {"x-native-latency": "12ms", "content-type": "application/json"},
        "chunks": [
            {"id": "1", "usage": {"prompt_tokens": 10}, "choices": [{"text": "Hello", "logprob": -0.5}]},
            {"id": "2", "usage": {"prompt_tokens": 10}, "choices": [{"text": " world", "logprob": -0.2}]},
        ],
    },
    {
        "headers": {"content-type": "application/json", "x-backend-id": "gpu-01"},
        "chunks": [
            {"meta": {"seq": 0}, "data": {"tokens": [101, 202]}},
            {"meta": {"seq": 1}, "data": {"tokens": [303]}},
        ],
    },
]

SHIM_RESPONSES = [
    {
        "headers": {"content-type": "application/json", "x-shim-version": "v1.2"},
        "chunks": [
            {"id": "1", "choices": [{"text": "Hello"}]},
            {"id": "2", "choices": [{"text": " world"}]},
        ],
    },
    {
        "headers": {"content-type": "application/json", "x-backend-id": "gpu-01"},
        "chunks": [
            {"meta": {"seq": 0}, "data": {"tokens": "101, 202"}},
            {"meta": {"seq": 1}, "data": {"tokens": "303"}},
            {"meta": {"seq": 2}, "data": {"tokens": ""}},
        ],
    },
]

EVENT_STREAMS = [
    [
        {"event": "request_start", "timestamp": 1000.000},
        {"event": "token", "timestamp": 1000.120, "id": 1},
        {"event": "token", "timestamp": 1000.150, "id": 2},
        {"event": "token", "timestamp": 1000.180, "id": 3},
        {"event": "token", "timestamp": 1000.220, "id": 4},
    ],
    [
        {"event": "init", "timestamp": 2000.000},
        {"token": "A", "timestamp": 2000.050},
        {"token": "B", "timestamp": 2000.100},
        {"token": "C", "timestamp": 2000.200},
    ],
]

BASE_PARAMS = {"temperature": 0.7, "top_k": 50, "repetition_penalty": 1.1}
PARAM_CANDIDATES = {
    "temperature": [0.2, 0.5, 0.9],
    "top_k": [10, 20, 100],
    "repetition_penalty": [1.0, 1.2, 1.5],
}


def mock_runner(params):
    t = params.get("temperature", 0.7)
    tp = params.get("top_k", 50)
    return {"sample": [t * 2, tp + 5]}


def diff_responses(native_resp, shim_resp):
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
