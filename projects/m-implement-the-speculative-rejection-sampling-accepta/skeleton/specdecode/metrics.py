import numpy as np


def compute_acceptance_probs(target_probs, draft_probs):
    """Compute per-position acceptance probability alpha_i."""
    raise NotImplementedError


def expected_accepted_tokens(acceptance_probs):
    """Compute expected number of accepted tokens given acceptance probabilities."""
    raise NotImplementedError
