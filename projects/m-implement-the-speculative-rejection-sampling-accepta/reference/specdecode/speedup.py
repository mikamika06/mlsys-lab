import numpy as np
from .metrics import expected_accepted_tokens


def calculate_speedup(acceptance_probs, draft_cost_ratio, verify_cost_ratio):
    """Estimate speedup factor relative to baseline standard decoding."""
    exp_tokens = expected_accepted_tokens(acceptance_probs)
    gamma = len(acceptance_probs)
    denom = gamma * draft_cost_ratio + verify_cost_ratio
    if denom <= 0.0:
        return 0.0
    return exp_tokens / denom
