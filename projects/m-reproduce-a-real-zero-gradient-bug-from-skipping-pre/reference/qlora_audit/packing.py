def compute_token_utilization(sequences, max_seq_len, pad_token_id, pack=False):
    if not sequences:
        return {"total_tokens": 0, "real_tokens": 0, "utilization_ratio": 0.0}

    if not pack:
        total_tokens = len(sequences) * max_seq_len
        real_tokens = sum(sum(1 for t in seq[:max_seq_len] if t != pad_token_id) for seq in sequences)
        ratio = real_tokens / total_tokens if total_tokens > 0 else 0.0
        return {
            "total_tokens": total_tokens,
            "real_tokens": real_tokens,
            "utilization_ratio": ratio,
        }

    all_real = [t for seq in sequences for t in seq if t != pad_token_id]
    packed_chunks = []
    current_chunk = []

    for token in all_real:
        if len(current_chunk) == max_seq_len:
            packed_chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(token)

    if current_chunk:
        while len(current_chunk) < max_seq_len:
            current_chunk.append(pad_token_id)
        packed_chunks.append(current_chunk)

    total_tokens = len(packed_chunks) * max_seq_len
    real_tokens = len(all_real)
    ratio = real_tokens / total_tokens if total_tokens > 0 else 0.0
    return {
        "total_tokens": total_tokens,
        "real_tokens": real_tokens,
        "utilization_ratio": ratio,
    }
