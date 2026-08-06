import hashlib
import json

CONFIGS = [
    {
        "model_name": "llama-7b",
        "precision": "fp16",
        "max_batch_size": 16,
        "max_seq_len": 2048,
        "plugins": ["rms_norm", "rotary_embedding"],
        "trt_version": "10.0.0"
    },
    {
        "model_name": "mistral-7b",
        "precision": "bf16",
        "max_batch_size": 32,
        "max_seq_len": 4096,
        "plugins": ["flash_attention"],
        "trt_version": "10.1.0"
    },
    {
        "model_name": "resnet-50",
        "precision": "fp32",
        "max_batch_size": 64,
        "max_seq_len": 1,
        "plugins": [],
        "trt_version": "9.3.0"
    }
]

def compute_cache_key(config):
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def optimal_queue_delay(arrival_rate, service_rate, max_batch_size, target_latency):
    best_delay = 0.001
    best_cost = float("inf")
    for d in [i * 0.001 for i in range(1, 100)]:
        avg_batch = min(max_batch_size, max(1.0, arrival_rate * d))
        latency = d + (avg_batch / service_rate)
        cost = abs(latency - target_latency) + (1.0 / avg_batch) * 0.1
        if cost < best_cost:
            best_cost = cost
            best_delay = d
    return round(best_delay, 4)

def decompose_cold_start(timings):
    total = sum(timings.values())
    if total <= 0:
        return {k: 0.0 for k in timings}
    return {k: round(v / total, 4) for k, v in timings.items()}
