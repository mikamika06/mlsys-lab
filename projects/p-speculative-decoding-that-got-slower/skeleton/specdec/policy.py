class AdaptivePolicy:
    def __init__(self, model, tracker, min_speedup=1.02):
        raise NotImplementedError

    def decide(self, domain, batch_size, max_gamma=8):
        raise NotImplementedError

    def simulate_request(self, domain, batch_size, gamma, accepted_count, base_step_time=10.0):
        raise NotImplementedError

    def evaluate_p95_and_throughput(self, traffic_stream):
        raise NotImplementedError
