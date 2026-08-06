def size_prefill_decode_ratio(prompt_len: int, output_len: int, prefill_ms_per_token: float = 0.5, decode_ms_per_token: float = 20.0) -> float:
    total_prefill_time = prompt_len * prefill_ms_per_token
    total_decode_time = output_len * decode_ms_per_token
    if total_decode_time == 0:
        return float('inf')
    return total_prefill_time / total_decode_time
