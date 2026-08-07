import numpy as np


class EagleEngine:
    def __init__(self, target_model_dim, vocab_size, draft_head):
        raise NotImplementedError

    def forward_target(self, x):
        raise NotImplementedError

    def generate_draft(self, hidden):
        raise NotImplementedError

    def verify(self, draft_tokens, target_logits, temperature=1.0):
        raise NotImplementedError

    def memory_usage_bytes(self, separate_draft_params=100_000_000):
        raise NotImplementedError
