import hashlib
import json

def compute_cache_key(req):
    canonical = json.dumps(req, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def compute_metrics(counters):
    hits = counters.get("hits", 0)
    misses = counters.get("misses", 0)
    hit_latency = counters.get("hit_latency_ms", 1.0)
    miss_latency = counters.get("miss_latency_ms", 50.0)
    bytes_per_entry = counters.get("bytes_per_entry", 1024)
    total_requests = hits + misses
    saved_latency = hits * (miss_latency - hit_latency)
    memory_cost = hits * bytes_per_entry
    return {
        "total_requests": total_requests,
        "saved_latency_ms": float(saved_latency),
        "memory_cost_bytes": int(memory_cost)
    }

def is_safe_config(cfg):
    if cfg.get("stochastic", False):
        return False
    if cfg.get("dynamic_state", False):
        return False
    return True

REQUESTS = [
    {"model": "llama", "prompt": "hello", "temperature": 0.0},
    {"model": "llama", "prompt": "hello", "temperature": 0.7},
    {"model": "falcon", "prompt": "world", "temperature": 0.0},
]

COUNTERS = [
    {"hits": 100, "misses": 50, "hit_latency_ms": 2.0, "miss_latency_ms": 40.0, "bytes_per_entry": 2048},
    {"hits": 200, "misses": 10, "hit_latency_ms": 1.5, "miss_latency_ms": 45.0, "bytes_per_entry": 1024},
]

CONFIGS = [
    {"model": "llama", "stochastic": False, "dynamic_state": False},
    {"model": "gpt", "stochastic": True, "dynamic_state": False},
    {"model": "mistral", "stochastic": False, "dynamic_state": True},
]
