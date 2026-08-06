"""Statistical calculations for benchmark sample size and confidence intervals."""
import numpy as np


def required_sample_size(samples, target_rel_ci=0.05, confidence=0.95):
    """Calculates the minimum required sample size to reach the target relative confidence interval width."""
    raise NotImplementedError


def compute_ci_bounds(samples, confidence=0.95):
    """Computes mean, standard error, and relative CI error bound for a sample array."""
    raise NotImplementedError
