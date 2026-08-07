import numpy as np


class TreeVerifier:
    def __init__(self, target_model):
        raise NotImplementedError

    def make_attention_mask(self, tree):
        raise NotImplementedError

    def verify(self, prefix, tree, target_logits):
        raise NotImplementedError
