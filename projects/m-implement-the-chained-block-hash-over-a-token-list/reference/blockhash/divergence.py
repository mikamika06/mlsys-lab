from blockhash.metrics import count_reusable_blocks

def find_divergence_token(tokens_a, tokens_b, block_size=16):
    reusable = count_reusable_blocks(tokens_a, tokens_b, block_size)
    div_token_idx = reusable * block_size
    total_blocks_b = (len(tokens_b) + block_size - 1) // block_size
    lost_count = max(0, total_blocks_b - reusable)
    return div_token_idx, lost_count
