def crossover_point(seq_len, d_model, heads, num_devices):
    bytes_per_elem = 2
    mono_mem = 2 * seq_len * d_model * bytes_per_elem
    ring_mem = 2 * (seq_len / num_devices) * d_model * bytes_per_elem + 2 * (seq_len / num_devices) * d_model * bytes_per_elem
    return ring_mem
