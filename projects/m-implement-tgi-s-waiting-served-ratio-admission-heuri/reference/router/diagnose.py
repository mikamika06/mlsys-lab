def diagnose_oom(logs: list, max_batch_prefill_tokens: int, max_batch_total_tokens: int) -> dict:
    """Identify the exact constraint violation causing an OOM from router logs."""
    for step_idx, entry in enumerate(logs):
        p_tok = entry.get("prefill_tokens", 0)
        t_tok = entry.get("total_tokens", 0)
        is_oom = entry.get("oom_occurred", False)

        exceeded_prefill = p_tok > max_batch_prefill_tokens
        exceeded_total = t_tok > max_batch_total_tokens

        if is_oom or exceeded_prefill or exceeded_total:
            if exceeded_prefill and not exceeded_total:
                cause = "EXCEEDED_MAX_BATCH_PREFILL_TOKENS"
            elif exceeded_total and not exceeded_prefill:
                cause = "EXCEEDED_MAX_BATCH_TOTAL_TOKENS"
            elif exceeded_prefill and exceeded_total:
                cause = "EXCEEDED_BOTH_LIMITS"
            else:
                cause = "UNKNOWN_OOM"

            return {
                "failed_step": step_idx,
                "cause": cause,
                "prefill_tokens": p_tok,
                "total_tokens": t_tok,
            }

    return {"failed_step": -1, "cause": "NO_OOM", "prefill_tokens": 0, "total_tokens": 0}
