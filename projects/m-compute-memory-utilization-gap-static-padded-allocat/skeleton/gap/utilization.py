def calculate_static_memory(seq_lengths, max_seq_len, bytes_per_token):
    raise NotImplementedError


def calculate_paged_memory(seq_lengths, block_size, bytes_per_token):
    raise NotImplementedError


def compute_utilization_gap(seq_lengths, max_seq_len, block_size, bytes_per_token):
    raise NotImplementedError
