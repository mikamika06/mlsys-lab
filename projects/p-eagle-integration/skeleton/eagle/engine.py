class EagleEngine:
    def __init__(self, base_model, draft_head):
        raise NotImplementedError

    def step(self, input_ids, temperature=1.0):
        raise NotImplementedError

    def measure_acceptance(self, reference_tokens, drafted_tokens):
        raise NotImplementedError

    def estimate_memory_bytes(self, separate_draft_params):
        raise NotImplementedError

    def compute_speedup(self, baseline_time, speculative_time):
        raise NotImplementedError
