"""Efficiency metrics and performance diagnostics for speculative decoding."""


def compute_breakeven_acceptance_rate(cost_ratio: float, gamma: int) -> float:
    """Calculate the minimum acceptance rate required for speculative speedup > 1.0."""
    if cost_ratio <= 0.0 or gamma <= 0:
        return 0.0
    val = (gamma * cost_ratio) / (1.0 + gamma * cost_ratio)
    return float(min(1.0, max(0.0, val)))


def analyze_prefill_speculation(prompt_length: int, gamma: int, draft_t_per_tok: float, target_t_batch: float) -> dict:
    """Measure latency components for prompt processing with and without speculation."""
    base_prefill_latency = target_t_batch
    draft_prefill_latency = prompt_length * draft_t_per_tok
    speculative_prefill_latency = base_prefill_latency + draft_prefill_latency
    speedup = base_prefill_latency / speculative_prefill_latency if speculative_prefill_latency > 0 else 0.0

    return {
        "base_latency": base_prefill_latency,
        "speculative_latency": speculative_prefill_latency,
        "speedup": speedup,
        "helps_prefill": speedup > 1.0,
        "reason": "Prefill is compute-bound parallel matrix multiplication; draft overhead adds sequential latency without reducing target work."
    }
