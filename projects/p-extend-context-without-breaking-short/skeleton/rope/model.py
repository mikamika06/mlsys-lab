import numpy as np
from rope.scaling import RoPEScaling

class ContextModel:
    def __init__(self, dim, scale_type="linear", factor=1.0):
        raise NotImplementedError

    def evaluate_perplexity(self, tokens):
        raise NotImplementedError

    def retrieve_needle(self, context, needle):
        raise NotImplementedError
