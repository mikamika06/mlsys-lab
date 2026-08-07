ADAPTERS = ["adapter_1", "adapter_2", "adapter_3", "adapter_4"]

def get_sample_requests():
    return [
        {"id": 1, "adapter_id": "adapter_1", "tokens": 32},
        {"id": 2, "adapter_id": "adapter_2", "tokens": 32},
        {"id": 3, "adapter_id": "adapter_3", "tokens": 32},
        {"id": 4, "adapter_id": "adapter_4", "tokens": 32},
    ]

def run_base_baseline(requests):
    total_tokens = sum(r.get("tokens", 1) for r in requests)
    return {"throughput": float(total_tokens), "total_tokens": total_tokens, "duration": 1.0}
