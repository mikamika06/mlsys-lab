def decode_step_flops(prompt_len: int, model_cfg: dict) -> int:
    """Compute FLOPs per decode step for a request."""
    raise NotImplementedError


def decode_step_time_ms(prompt_len: int, model_cfg: dict, tflops: float) -> float:
    """Calculate decode step compute time in milliseconds."""
    raise NotImplementedError


def compute_pd_ratio(prefill_ms: float, transfer_ms: float, decode_step_ms: float, gen_tokens: int) -> float:
    """Calculate target P:D replica ratio accounting for transfer latency."""
    raise NotImplementedError


def allocate_pd_nodes(total_nodes: int, prefill_total_ms: float, decode_total_ms: float) -> tuple[int, int]:
    """Find optimal integer allocation of prefill and decode nodes."""
    raise NotImplementedError
