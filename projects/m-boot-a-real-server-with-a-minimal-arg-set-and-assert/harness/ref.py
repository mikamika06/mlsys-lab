TEST_CASES = [
    {
        "args": [
            "--model",
            "meta-llama/Llama-2-7b-hf",
            "--served-model-name",
            "llama2-7b",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
        ],
        "expected_model": "meta-llama/Llama-2-7b-hf",
        "expected_served_name": "llama2-7b",
        "expected_host": "127.0.0.1",
        "expected_port": 8080,
    },
    {
        "args": ["--model", "Qwen/Qwen2.5-7B-Instruct"],
        "expected_model": "Qwen/Qwen2.5-7B-Instruct",
        "expected_served_name": "Qwen/Qwen2.5-7B-Instruct",
        "expected_host": "0.0.0.0",
        "expected_port": 8000,
    },
    {
        "args": [
            "--model",
            "mistralai/Mistral-7B-v0.1",
            "--served-model-name",
            "mistral-fast",
            "--port",
            "9090",
        ],
        "expected_model": "mistralai/Mistral-7B-v0.1",
        "expected_served_name": "mistral-fast",
        "expected_host": "0.0.0.0",
        "expected_port": 9090,
    },
]
