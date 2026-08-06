"""Speculative decoding throughput evaluation."""

from specquant.draft import compute_acceptance_rate, compute_draft_latency


def expected_accepted_tokens(alpha, draft_len):
    """Computes expected accepted tokens per speculative iteration."""
    if abs(alpha - 1.0) < 1e-9:
        return float(draft_len + 1)
    return (1.0 - (alpha ** (draft_len + 1))) / (1.0 - alpha)


def compute_speculative_throughput(params_m, precision, draft_len, system_config, alpha_config):
    """Computes end-to-end token generation throughput (tokens/sec)."""
    t_draft_step = compute_draft_latency(params_m, precision, system_config)
    t_target = system_config["target_step_ms"]
    total_iteration_ms = draft_len * t_draft_step + t_target
    total_iteration_sec = total_iteration_ms / 1000.0

    alpha = compute_acceptance_rate(
        params_m,
        precision,
        alpha_config["base_alpha_max"],
        alpha_config["alpha_scale_m"]
    )
    n_accepted = expected_accepted_tokens(alpha, draft_len)
    return n_accepted / total_iteration_sec


def compute_throughput_ratio(params_m, draft_len, system_config, alpha_config):
    """Computes ratio of INT8 throughput to FP16 throughput."""
    tps_int8 = compute_speculative_throughput(params_m, "int8", draft_len, system_config, alpha_config)
    tps_fp16 = compute_speculative_throughput(params_m, "fp16", draft_len, system_config, alpha_config)
    return tps_int8 / tps_fp16
