PARAMS = (
    "frequency_penalty", "logit_bias", "logprobs", "max_tokens", "n",
    "presence_penalty", "response_format", "seed", "stop", "stream",
    "temperature", "tools", "top_logprobs", "top_p", "user",
)


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
