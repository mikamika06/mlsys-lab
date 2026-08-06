CONFIGS = [
    {"seq_lens": [128, 256, 512], "block_size": 16, "num_layers": 32, "block_tables": [[0, 1], [2, 3, 4], [4, 5]]},
    {"seq_lens": [64, 128], "block_size": 32, "num_layers": 16, "block_tables": [[10, 11], [11, 12]]},
    {"seq_lens": [1024, 2048, 512, 256], "block_size": 16, "num_layers": 24, "block_tables": [[0, 1, 2], [3, 4], [5, 6, 7, 8]]},
]

def compute_budget(seq_lens, block_size, num_layers):
    total = 0
    for length in seq_lens:
        blocks = (length + block_size - 1) // block_size
        total += blocks * num_layers
    return total

def measure_allocated_blocks(block_tables, block_size):
    unique_blocks = set()
    for table in block_tables:
        for block_id in table:
            if block_id >= 0:
                unique_blocks.add(block_id)
    return len(unique_blocks)

def compute_relative_error(actual, predicted):
    if predicted == 0:
        return 0.0 if actual == 0 else 1.0
    return abs(actual - predicted) / float(predicted)
