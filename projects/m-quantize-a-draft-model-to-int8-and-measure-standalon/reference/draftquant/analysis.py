import numpy as np


def evaluate_acceptance_change(baseline_rates: np.ndarray, quantized_rates: np.ndarray) -> float:
    diff = np.mean(quantized_rates) - np.mean(baseline_rates)
    return float(diff)


def is_net_end_to_end_win(latency_ratio: float, acceptance_rate_ratio: float, target_cost: float, draft_cost: float) -> bool:
    baseline_step_cost = draft_cost + target_cost
    quant_draft_cost = draft_cost * latency_ratio
    quant_expected_accepted = acceptance_rate_ratio
    effective_quant_cost = quant_draft_cost + (1.0 - quant_expected_accepted) * target_cost
    return bool(effective_quant_cost < baseline_step_cost)
