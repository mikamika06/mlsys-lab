CASES = [
    {
        "tokens": ["hello", "world", "!", "!", "!", "!", "!"],
        "expected": True
    },
    {
        "tokens": ["once", "upon", "a", "time", "there", "was"],
        "expected": False
    },
    {
        "tokens": ["ok", "ok", "ok", "ok", "ok", "ok"],
        "expected": True
    }
]

BATCH_DATA = {
    "requests": [
        {"id": "req_101", "index": 0, "max_retries": 3, "retry_count": 0},
        {"id": "req_102", "index": 1, "max_retries": 2, "retry_count": 2},
        {"id": "req_103", "index": 2, "max_retries": 3, "retry_count": 0}
    ],
    "crash_index": 1
}
