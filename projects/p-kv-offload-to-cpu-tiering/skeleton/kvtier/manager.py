class TierManager:
    def __init__(self, gpu_capacity, cpu_capacity):
        raise NotImplementedError

    def offload(self, session_id):
        raise NotImplementedError

    def prefetch(self, session_id):
        raise NotImplementedError

    def get_p95_latency(self):
        raise NotImplementedError
