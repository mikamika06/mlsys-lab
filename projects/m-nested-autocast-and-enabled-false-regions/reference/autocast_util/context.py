_GLOBAL_STACK = []


class NestedAutocastManager:
    def __init__(self, enabled=True, dtype="fp16"):
        self.enabled = enabled
        self.dtype = dtype
        self.explicit_disable = not enabled

    def __enter__(self):
        _GLOBAL_STACK.append({
            "enabled": self.enabled,
            "dtype": self.dtype,
            "explicit_disable": self.explicit_disable
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _GLOBAL_STACK:
            _GLOBAL_STACK.pop()
