import random

random.seed(42)


def generate_requests():
    return [
        {"id": i, "prefix": [1, 2, i % 3], "tokens": [1, 2, i % 3, i]}
        for i in range(5)
    ]


def reorder_requests(requests, budget):
    return sorted(requests, key=lambda r: (-len(r.get("prefix", [])), r.get("id", 0)))


def identify_eviction(requests, cache_budget):
    cached = set()
    for r in requests:
        pref = set(r.get("prefix", []))
        if len(cached.union(pref)) > cache_budget:
            return r.get("id")
        cached.update(pref)
    return -1


def apc_ttft_ratio(prompt_len, batch_size, apc_on):
    base_ttft = 50.0 if apc_on else 200.0
    return float(base_ttft + 2.0 * batch_size + (prompt_len / 1024.0))


CONFIGS = [generate_requests() for _ in range(3)]
