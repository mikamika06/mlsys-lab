import numpy as np


class GrammarFSM:
    def __init__(self, vocab_size, transitions, start_state=0):
        self.vocab_size = vocab_size
        self.transitions = transitions
        self.start_state = start_state

    def mask_logits(self, state, logits):
        masked = np.copy(logits)
        allowed = self.transitions.get(state, {})
        allowed_indices = set(allowed.keys())
        mask = np.full(self.vocab_size, -1e9, dtype=logits.dtype)
        for idx in allowed_indices:
            if idx < self.vocab_size:
                mask[idx] = 0.0
        return masked + mask

    def step(self, state, token):
        return self.transitions.get(state, {}).get(token, state)
