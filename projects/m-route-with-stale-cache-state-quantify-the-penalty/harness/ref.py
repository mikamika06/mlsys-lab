import random

def get_scenarios():
    random.seed(42)
    scenarios = []
    for i in range(5):
        req = [random.randint(1, 10) for _ in range(32)]
        workers = []
        for j in range(3):
            cached = [random.randint(1, 10) for _ in range(32)]
            workers.append({
                "worker_id": f"worker_{j}",
                "cached_tokens": cached,
                "last_access_tick": random.randint(0, 100),
                "decay_factor": 0.02
            })
        scenarios.append({
            "request_tokens": req,
            "workers": workers,
            "transfer_cost": 1.5,
            "compute_cost": 4.0,
            "current_tick": 110
        })
    return scenarios
