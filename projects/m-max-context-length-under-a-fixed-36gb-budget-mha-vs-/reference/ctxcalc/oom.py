def back_calculate_oom(failed_len, failed_bytes, budget_bytes=36 * 1024 * 1024 * 1024):
    return int(failed_len * (budget_bytes / failed_bytes))
