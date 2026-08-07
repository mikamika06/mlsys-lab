def profile_latency(batch_size, seq_len):
    return 10.0 + 0.5 * batch_size + 0.1 * seq_len + 0.05 * batch_size * seq_len


def warmup_cost(batch_sizes, seq_lens):
    return sum(profile_latency(b, s) for b in batch_sizes for s in seq_lens)
