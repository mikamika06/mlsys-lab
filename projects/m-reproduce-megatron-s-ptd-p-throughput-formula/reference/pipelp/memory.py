def peak_memory_1f1b(pp_size, microbatches, hidden_size, seq_len, batch_size):
    base = batch_size * seq_len * hidden_size * 4
    return int(base * (pp_size + microbatches))


def peak_memory_interleaved(pp_size, microbatches, virtual_pp_stages, hidden_size, seq_len, batch_size):
    base = batch_size * seq_len * hidden_size * 4
    reduction = float(virtual_pp_stages)
    return int(base * (pp_size + microbatches / reduction))
