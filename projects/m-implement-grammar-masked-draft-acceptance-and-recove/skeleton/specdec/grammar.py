import numpy as np


class GrammarFSM:
    def __init__(self, vocab_size, transitions, start_state=0):
        raise NotImplementedError

    def mask_logits(self, state, logits):
        raise NotImplementedError

    def step(self, state, token):
        raise NotImplementedError
