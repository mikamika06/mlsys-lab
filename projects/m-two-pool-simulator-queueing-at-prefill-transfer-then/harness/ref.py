CONFIGS = [
    {
        "kv_connector": "NixlConnector",
        "roles": [
            {"rank": 0, "role": "prefill"},
            {"rank": 1, "role": "decode"}
        ]
    },
    {
        "kv_connector": "NixlConnector",
        "roles": [
            {"rank": 0, "role": "prefill"},
            {"rank": 1, "role": "prefill"},
            {"rank": 2, "role": "decode"}
        ]
    }
]

INVALID_CONFIGS = [
    {"kv_connector": "NixlConnector", "roles": [{"rank": 0, "role": "prefill"}, {"rank": 0, "role": "decode"}]},
    {"kv_connector": "NixlConnector", "roles": [{"rank": 1, "role": "prefill"}]},
    {"roles": [{"rank": 0, "role": "prefill"}]},
    "not-a-dict"
]

TOPOLOGY = {
    "nodes": [
        {"id": "n1", "role": "prefill", "tier": "core"},
        {"id": "n2", "role": "decode", "tier": "core"},
        {"id": "n3", "role": "decode", "tier": "edge"}
    ]
}

REQUESTS = [
    {"id": "req-1", "prompt_tokens": 100, "kv_size_bytes": 1024, "decode_tokens": 10}
]
