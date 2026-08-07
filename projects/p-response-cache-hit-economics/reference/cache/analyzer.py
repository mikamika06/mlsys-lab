"""Trace analysis and memory accounting for prompt response caching."""


def analyze_trace(trace):
    if not trace:
        return {"total_requests": 0, "unique_keys": 0, "hits": 0, "hit_rate": 0.0}
    seen = set()
    hits = 0
    for req in trace:
        key = req["key"]
        if key in seen:
            hits += 1
        else:
            seen.add(key)
    total = len(trace)
    return {
        "total_requests": total,
        "unique_keys": len(seen),
        "hits": hits,
        "hit_rate": hits / total if total > 0 else 0.0,
    }


def calculate_memory_footprint(num_entries, avg_key_len_tokens, avg_val_len_tokens, bytes_per_token=2):
    bytes_per_entry = (avg_key_len_tokens + avg_val_len_tokens) * bytes_per_token
    total_bytes = num_entries * bytes_per_entry
    return {
        "bytes_per_entry": bytes_per_entry,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
    }
