import hashlib
import json
import math
import numpy as np


def generate_deployment_matrix():
    gpus = ["A100", "H100", "L40S"]
    precisions = ["fp16", "int8", "fp8"]
    cuda_versions = ["12.1", "12.4"]
    specs = []
    for g in gpus:
        for p in precisions:
            for c in cuda_versions:
                specs.append({
                    "gpu": g,
                    "precision": p,
                    "cuda_version": c,
                    "opt_shapes": {"input": [1, 128, 768]}
                })
    return specs


def ref_resolve_matrix(specs):
    store = {}
    hits = 0
    misses = 0
    resolved = []
    for idx, s in enumerate(specs):
        norm = json.dumps(s, sort_keys=True)
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h in store:
            hits += 1
            resolved.append(store[h])
        else:
            misses += 1
            art_id = f"engine_art_{idx}"
            store[h] = art_id
            resolved.append(art_id)
    return store, resolved, hits, misses


def ref_calculate_instances(arrival_rate_rps, mean_service_time_ms, p99_service_time_ms, target_p99_slo_ms, max_utilization=0.85):
    service_rate_per_instance = 1000.0 / mean_service_time_ms
    min_instances_capacity = math.ceil(arrival_rate_rps / (service_rate_per_instance * max_utilization))

    for instances in range(max(1, min_instances_capacity), 10000):
        total_capacity = instances * service_rate_per_instance
        rho = arrival_rate_rps / total_capacity
        if rho >= 1.0:
            continue
        queue_wait_ms = (rho / (1.0 - rho)) * mean_service_time_ms / instances
        estimated_p99_ms = p99_service_time_ms + 2.326 * queue_wait_ms
        if estimated_p99_ms <= target_p99_slo_ms and rho <= max_utilization:
            return instances
    return min_instances_capacity
