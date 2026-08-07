def find_optimal_block_size(lengths, block_sizes, bytes_per_entry=4):
    raise NotImplementedError


def evaluate_trace(block_size, trace):
    raise NotImplementedError


def check_threshold(block_size, lengths, max_loss_ratio):
    raise NotImplementedError


def recommend_block_size(lengths_distribution):
    raise NotImplementedError
