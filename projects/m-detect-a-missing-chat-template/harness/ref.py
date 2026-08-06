CONFIGS = [
    {"chat_template": "{% if messages %}{{ messages[0]['content'] }}{% endif %}"},
    {"chat_template": None},
    {"chat_template": ""},
    {"chat_template": "plain text without tags"},
    {"chat_template": "{{ user_input }}"}
]

OPTIMIZATION_TESTS = [
    {"target_tokens": 10000, "max_seqlen": 2048, "available_memory_mb": 512},
    {"target_tokens": 50000, "max_seqlen": 4096, "available_memory_mb": 1024},
    {"target_tokens": 2000, "max_seqlen": 512, "available_memory_mb": 256},
]

HESSIAN_TESTS = [
    {"hidden_size": 2048, "num_parameters": 500000, "dtype_bytes": 2},
    {"hidden_size": 4096, "num_parameters": 1000000, "dtype_bytes": 4},
]
