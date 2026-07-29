OPENAI_USAGE_FIELDS = ("completion_tokens", "prompt_tokens", "total_tokens")


def native_counters(runner):
    return sorted(set(runner["native_counters"]))


def shim_counters(runner):
    return sorted(set(runner["native_counters"]) & set(OPENAI_USAGE_FIELDS))


def hidden_counters(runner):
    return sorted(set(native_counters(runner)) - set(shim_counters(runner)))
