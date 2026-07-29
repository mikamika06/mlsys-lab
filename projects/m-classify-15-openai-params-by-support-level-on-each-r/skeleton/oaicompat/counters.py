OPENAI_USAGE_FIELDS = ("completion_tokens", "prompt_tokens", "total_tokens")


def native_counters(runner):
    raise NotImplementedError


def shim_counters(runner):
    raise NotImplementedError


def hidden_counters(runner):
    raise NotImplementedError
