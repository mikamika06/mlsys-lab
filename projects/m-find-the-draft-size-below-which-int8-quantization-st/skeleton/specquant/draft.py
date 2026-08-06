"""Draft model step latency and acceptance rate modeling."""


def compute_draft_latency(params_m, precision, system_config):
    """Computes single-step latency for a draft model in milliseconds."""
    raise NotImplementedError


def compute_acceptance_rate(params_m, precision, base_alpha_max, alpha_scale_m):
    """Computes expected acceptance rate alpha for a given draft model size."""
    raise NotImplementedError
