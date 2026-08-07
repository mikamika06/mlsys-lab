def apc_ttft_ratio(prompt_len, batch_size, apc_on):
    base_ttft = 50.0 if apc_on else 200.0
    return float(base_ttft + 2.0 * batch_size + (prompt_len / 1024.0))
