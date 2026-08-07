class EagleIntegration:
    def __init__(self, target_model, draft_head):
        raise NotImplementedError

    def measure_acceptance(self, prompts):
        raise NotImplementedError

    def estimate_memory_mb(self):
        raise NotImplementedError

    def compute_speedup(self, baseline_time, speculative_time):
        raise NotImplementedError
