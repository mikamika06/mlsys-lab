import random

def generate_sweep_data():
    rng = random.Random(42)
    data = []
    for c in [1, 2, 4, 8, 16, 32, 64, 128]:
        lat = 20.0 + c * 1.5 + rng.uniform(-2.0, 2.0)
        data.append({"concurrency": c, "p99_latency_ms": lat})
    return data

def find_max_concurrency(data, sla_ms):
    valid = [d["concurrency"] for d in data if d["p99_latency_ms"] <= sla_ms]
    return max(valid) if valid else 0

def generate_latency_runs():
    return {
        "batch_off": {"queue_ms": 1.0, "compute_ms": 15.0, "total_ms": 16.0},
        "batch_on": {"queue_ms": 8.0, "compute_ms": 12.0, "total_ms": 20.0}
    }

def attribute_delta(off_run, on_run):
    dq = on_run["queue_ms"] - off_run["queue_ms"]
    dc = on_run["compute_ms"] - off_run["compute_ms"]
    dt = on_run["total_ms"] - off_run["total_ms"]
    return {"queue_delta": dq, "compute_delta": dc, "total_delta": dt}

def generate_throughput_fixture():
    return {
        "total_tokens": 10000,
        "gpu_count": 2,
        "active_users": 50,
        "total_time_sec": 10.0
    }

def compute_tokens_metrics(fixture):
    t = fixture["total_time_sec"]
    tok = fixture["total_tokens"]
    gpus = fixture["gpu_count"]
    users = fixture["active_users"]
    return {
        "tokens_per_sec_gpu": (tok / t) / gpus,
        "tokens_per_sec_user": (tok / t) / users
    }
