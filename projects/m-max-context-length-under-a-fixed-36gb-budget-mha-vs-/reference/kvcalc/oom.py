from kvcalc.memory import compute_kv_bytes
from kvcalc.budget import max_context_length

BUDGET_BYTES = 36 * 1024 * 1024 * 1024


def back_calculate_oom(config, failed_context_length, batch_size=1):
    failed_bytes = compute_kv_bytes(config, failed_context_length, batch_size)
    if failed_bytes <= BUDGET_BYTES:
        raise ValueError("Did not actually OOM under budget")
    return max_context_length(config, BUDGET_BYTES, batch_size)
