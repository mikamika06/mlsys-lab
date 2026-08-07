def compute_latency_ratio(seq_len, batch_size, hidden_dim, is_stateful):
    base = float(seq_len * hidden_dim * batch_size)
    if is_stateful:
        return base * 0.8
    else:
        return base * 1.5
