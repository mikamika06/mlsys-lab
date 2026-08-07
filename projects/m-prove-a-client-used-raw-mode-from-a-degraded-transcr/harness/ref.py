TRANSCRIPTS = [
    ("GET /v1/completions\nHost: localhost\nX-Raw-Mode: true\n[RAW_EXECUTION]\n", True),
    ("GET /v1/completions\nHost: localhost\n", False),
    ("POST /api/chat\nmode=raw\n[RAW_EXECUTION]\n", True),
]

CONTEXTS = [
    ([], [{"role": "user", "content": "hello"}], [{"role": "user", "content": "hello", "state_carried": True}]),
    ([{"role": "user", "content": "prev"}], [{"role": "user", "content": "next"}], [{"role": "user", "content": "prev"}, {"role": "user", "content": "next", "state_carried": True}]),
]

REQUESTS = [
    ({"temperature": 0.2, "max_tokens": 32, "messages": [{"role": "user", "content": "test"}]}, {"prompt": "user: test", "temperature": 0.2, "max_tokens": 32, "stream": False}),
]
