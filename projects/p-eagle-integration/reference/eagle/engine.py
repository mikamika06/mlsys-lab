import numpy as np


class EagleEngine:
    def __init__(self, base_model, draft_head):
        self.base_model = base_model
        self.draft_head = draft_head

    def step(self, input_ids, temperature=1.0):
        hidden = self.base_model(input_ids)
        logits = self.draft_head.forward(hidden)
        return logits

    def measure_acceptance(self, reference_tokens, drafted_tokens):
        matches = 0
        total = len(reference_tokens)
        if total == 0:
            return 0.0
        for r, d in zip(reference_tokens, drafted_tokens):
            if r == d:
                matches += 1
        return float(matches / total)

    def estimate_memory_bytes(self, separate_draft_params):
        integrated_overhead = 1024 * 1024
        separate_memory = separate_draft_params * 4 + 50 * 1024 * 1024
        return float(separate_memory - integrated_overhead)

    def compute_speedup(self, baseline_time, speculative_time):
        if speculative_time <= 0:
            return 1.0
        return float(baseline_time / speculative_time)
