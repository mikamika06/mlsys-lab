EVENTS = [
    [{"type": "push", "device": "cuda", "dtype": "float16", "enabled": True},
     {"type": "push", "device": "cuda", "dtype": "float16", "enabled": False},
     {"type": "pop"},
     {"type": "pop"}],
    [{"type": "push", "device": "cuda", "dtype": "bfloat16", "enabled": True},
     {"type": "push", "device": "cuda", "dtype": "float16", "enabled": True},
     {"type": "pop"}],
    [{"type": "push", "device": "cuda", "dtype": "float16", "enabled": False},
     {"type": "push", "device": "cuda", "dtype": "float16", "enabled": True},
     {"type": "pop"},
     {"type": "pop"}],
    [{"type": "push", "device": "cuda", "dtype": "float16", "enabled": True}],
    [{"type": "push", "device": "cuda", "dtype": "bfloat16", "enabled": False}]
]
