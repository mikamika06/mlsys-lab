import numpy as np


def scrape_load(server_state, client_requests):
    rng = np.random.default_rng(42)
    latencies = []
    for req in client_requests:
        queue_delay = rng.exponential(scale=server_state.get("load_factor", 1.0))
        exec_time = req.get("tokens", 10) * 2.5
        total_lat = queue_delay + exec_time + rng.normal(0, 1.0)
        latencies.append(max(0.1, total_lat))
    return {
        "latencies": sorted(latencies),
        "total_requests": len(client_requests),
        "server_snapshots": server_state.get("snapshots", [])
    }
