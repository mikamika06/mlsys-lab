"""Speculative decoding throughput evaluation."""


def expected_accepted_tokens(alpha, draft_len):
    """Computes expected accepted tokens per speculative iteration."""
    raise NotImplementedError


def compute_speculative_throughput(params_m, precision, draft_len, system_config, alpha_config):
    """Computes end-to-end token generation throughput (tokens/sec)."""
    raise NotImplementedError


def compute_throughput_ratio(params_m, draft_len, system_config, alpha_config):
    """Computes ratio of INT8 throughput to FP16 throughput."""
    raise NotImplementedError
