"""Efficiency metrics and performance diagnostics for speculative decoding."""


def compute_breakeven_acceptance_rate(cost_ratio: float, gamma: int) -> float:
    """Calculate the minimum acceptance rate required for speculative speedup > 1.0."""
    raise NotImplementedError


def analyze_prefill_speculation(prompt_length: int, gamma: int, draft_t_per_tok: float, target_t_batch: float) -> dict:
    """Measure latency components for prompt processing with and without speculation."""
    raise NotImplementedError
