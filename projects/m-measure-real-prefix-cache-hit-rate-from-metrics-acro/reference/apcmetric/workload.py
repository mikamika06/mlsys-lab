class MockClient:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.idx = 0

    def get_metrics(self):
        s = self.snapshots[min(self.idx, len(self.snapshots) - 1)]
        self.idx += 1
        return s

def compute_hit_rate(client, workload_fn):
    initial = client.get_metrics()
    workload_fn()
    final = client.get_metrics()

    hits_key = "vllm:gpu_cache_config_prefix_cache_hit_total"
    requests_key = "vllm:num_requests_waiting"

    h_init = initial.get(hits_key, 0.0)
    h_final = final.get(hits_key, 0.0)

    hits = h_final - h_init

    total_key = "vllm:prompt_tokens_total"
    t_init = initial.get(total_key, 0.0)
    t_final = final.get(total_key, 0.0)
    total = t_final - t_init

    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, hits / max(1.0, total)))

def run_two_turn_workload(client, turn_fn):
    turn_fn(1)
    turn_fn(2)
    return True
