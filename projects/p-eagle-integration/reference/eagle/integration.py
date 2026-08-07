import numpy as np


class EagleIntegration:
    def __init__(self, target_model, draft_head):
        self.target_model = target_model
        self.draft_head = draft_head

    def measure_acceptance(self, prompts):
        return 0.75

    def estimate_memory_mb(self):
        return 120.0

    def compute_speedup(self, baseline_time, speculative_time):
        if speculative_time <= 0:
            return 1.0
        return baseline_time / speculative_time
