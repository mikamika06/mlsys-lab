def predict_blocks(seq_len, block_size, max_tokens):
    total_len = seq_len + max_tokens
    return (total_len + block_size - 1) // block_size
