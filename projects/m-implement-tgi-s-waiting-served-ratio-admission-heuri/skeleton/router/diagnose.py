def diagnose_oom(logs: list, max_batch_prefill_tokens: int, max_batch_total_tokens: int) -> dict:
    """Identify the exact constraint violation causing an OOM from router logs."""
    raise NotImplementedError
