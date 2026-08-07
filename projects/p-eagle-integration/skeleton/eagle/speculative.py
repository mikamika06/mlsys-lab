import numpy as np


class SpeculativeEngine:
    def __init__(self, target_model, draft_head, vocab_size):
        raise NotImplementedError

    def generate_draft(self, hidden_states, k):
        raise NotImplementedError

    def verify_and_sample(self, target_logits, draft_logits, draft_tokens, temperature):
        raise NotImplementedError

    def compute_memory(self, separate_draft_params, head_params):
        raise NotImplementedError

    def measure_speedup(self, base_time, speculative_time):
        raise NotImplementedError
