def calculate_internal_fragmentation(seq_lengths, block_size):
    raise NotImplementedError


def calculate_prefix_truncation_loss(shared_prefixes, request_prefixes, block_size):
    raise NotImplementedError


def evaluate_workload_objective(seq_lengths, shared_prefixes, request_prefixes, block_size, hit_penalty_weight):
    raise NotImplementedError
