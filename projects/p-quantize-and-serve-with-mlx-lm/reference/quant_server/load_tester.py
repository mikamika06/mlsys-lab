import time
import numpy as np

def simulate_load(server_fn, num_users: int = 3, requests_per_user: int = 10) -> dict:
    latencies = []
    for _ in range(num_users * requests_per_user):
        start = time.time()
        server_fn({"messages": [{"role": "user", "content": "hello"}]})
        latencies.append(time.time() - start)
    p95 = float(np.percentile(latencies, 95))
    return {"p95": p95, "success": True}
