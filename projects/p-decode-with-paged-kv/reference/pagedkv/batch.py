def build_block_table(seq_lens, block_size):
    tables = []
    for slen in seq_lens:
        n_blocks = max(1, (slen + block_size - 1) // block_size)
        tables.append(list(range(n_blocks)))
    return tables

def process_variable_batch(seq_lens, block_size):
    return build_block_table(seq_lens, block_size)
