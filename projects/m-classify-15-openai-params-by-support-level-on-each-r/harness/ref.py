PARAMS = (
    "frequency_penalty", "logit_bias", "logprobs", "max_tokens", "n",
    "presence_penalty", "response_format", "seed", "stop", "stream",
    "temperature", "tools", "top_logprobs", "top_p", "user",
)

OPENAI_USAGE_FIELDS = ("completion_tokens", "prompt_tokens", "total_tokens")

RUNNERS = [
    {
        "name": "gpu-pool",
        "supported": {"temperature", "top_p", "n", "stream", "stop", "max_tokens",
                      "presence_penalty", "frequency_penalty", "seed", "logprobs",
                      "top_logprobs", "tools"},
        "ignored": {"user"},
        "native_counters": ["prompt_tokens", "completion_tokens", "total_tokens",
                             "cache_hit_tokens", "queue_ms", "gpu_kv_blocks"],
    },
    {
        "name": "quantized-pool",
        "supported": {"temperature", "top_p", "stop", "max_tokens", "seed", "logit_bias"},
        "ignored": {"presence_penalty", "frequency_penalty", "n", "user"},
        "native_counters": ["prompt_tokens", "completion_tokens", "total_tokens",
                             "prefill_ms", "decode_ms"],
    },
    {
        "name": "cpu-fallback",
        "supported": {"temperature", "top_p", "stop", "max_tokens"},
        "ignored": {"seed", "n", "stream", "user"},
        "native_counters": ["prompt_tokens", "completion_tokens", "total_tokens",
                             "queue_ms", "batch_size"],
    },
    {
        "name": "edge-mini",
        "supported": {"temperature", "max_tokens", "stop"},
        "ignored": {"top_p", "user", "stream"},
        "native_counters": ["prompt_tokens", "completion_tokens", "total_tokens"],
    },
    {
        "name": "compat-pool",
        "supported": {"temperature", "top_p", "n", "stream", "stop", "max_tokens",
                      "presence_penalty", "frequency_penalty", "logit_bias", "seed",
                      "logprobs", "top_logprobs", "response_format", "tools", "user"},
        "ignored": set(),
        "native_counters": ["prompt_tokens", "completion_tokens", "total_tokens"],
    },
]


def classify_params(runner):
    supported = set(runner["supported"])
    ignored = set(runner["ignored"])
    out = []
    for name in sorted(PARAMS):
        if name in supported:
            level = "supported"
        elif name in ignored:
            level = "ignored"
        else:
            level = "unsupported"
        out.append({"param": name, "level": level})
    return out


def native_counters(runner):
    return sorted(set(runner["native_counters"]))


def shim_counters(runner):
    return sorted(set(runner["native_counters"]) & set(OPENAI_USAGE_FIELDS))


def hidden_counters(runner):
    return sorted(set(native_counters(runner)) - set(shim_counters(runner)))
