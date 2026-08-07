def streaming_llm_indices(seq_len, num_sinks, window_size):
    if seq_len <= 0:
        return []
    sinks = list(range(min(seq_len, num_sinks)))
    if seq_len <= num_sinks:
        return sorted(list(set(sinks)))
    start = max(num_sinks, seq_len - window_size)
    window = list(range(start, seq_len))
    return sorted(list(set(sinks + window)))
