OLLAMA_OPTIONS = [
    ("num_ctx", {"equivalent": "context_length", "supported": True}),
    ("temperature", {"equivalent": "temperature", "supported": True}),
    ("top_p", {"equivalent": "top_p", "supported": True}),
    ("num_predict", {"equivalent": "max_tokens", "supported": True}),
    ("repeat_penalty", {"equivalent": "repeat_penalty", "supported": True}),
    ("stop", {"equivalent": "stop", "supported": True}),
    ("seed", {"equivalent": "seed", "supported": True}),
    ("num_thread", {"equivalent": "cpu_threads", "supported": True}),
    ("tfs_z", {"equivalent": None, "supported": False}),
    ("typical_p", {"equivalent": None, "supported": False}),
]

TTL_SCENARIOS = [
    {"ttl": 300, "ticks": [0, 100, 250, 600], "expected_states": ["loaded", "loaded", "loaded", "evicted"]},
    {"ttl": 60, "ticks": [0, 50, 120], "expected_states": ["loaded", "loaded", "evicted"]},
]
