def get_test_benchmarks():
    return [
        {
            "id": "bm-normal",
            "total_tokens": 1000,
            "generation_duration": 5.0,
            "load_duration": 2.0,
            "reported_tok_s": 200.0
        },
        {
            "id": "bm-folded",
            "total_tokens": 1000,
            "generation_duration": 5.0,
            "load_duration": 2.0,
            "reported_tok_s": 142.85714285714286
        },
        {
            "id": "bm-edge",
            "total_tokens": 500,
            "generation_duration": 2.5,
            "load_duration": 0.0,
            "reported_tok_s": 200.0
        }
    ]
