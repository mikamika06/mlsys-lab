def peak_kv_transfer_bandwidth(
    prefill_complete_times: list,
    num_tokens: list,
    transfer_durations: list,
    kv_bytes_per_token: float,
) -> float:
    """
    Request i's KV cache (num_tokens[i] * kv_bytes_per_token bytes) is
    transferred at a CONSTANT rate over the half-open time window
    [prefill_complete_times[i], prefill_complete_times[i] + transfer_durations[i]):

        rate_i = (num_tokens[i] * kv_bytes_per_token) / transfer_durations[i]

    Return the peak aggregate bandwidth demanded at any single instant:
    max over t of the sum of rate_i over every request whose window
    covers t. A transfer's window is half-open, so a transfer ending at
    exactly t does NOT overlap one starting at exactly t.
    """
    raise NotImplementedError('your code here')
