def quantify_hol_blocking(long_prefill_len, num_short_decodes, short_decode_len, time_per_prefill_token, time_per_decode_step):
    prefill_time = long_prefill_len * time_per_prefill_token
    total_delay = prefill_time * num_short_decodes
    return float(total_delay)
