import numpy as np


def dispatch_tokens(tokens, indices, weights, num_experts, capacity):
    """
    Dispatch tokens to experts with capacity limits.
    """
    raise NotImplementedError


def combine_tokens(expert_outputs, dispatch_meta):
    """
    Combine expert outputs back to token sequence using weights.
    """
    raise NotImplementedError
