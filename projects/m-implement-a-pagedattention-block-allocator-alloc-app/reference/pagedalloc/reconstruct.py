import numpy as np


def reconstruct_sequence(allocator, block_table: list[int], seq_len: int) -> list[int]:
    tokens = []
    block_size = allocator.block_size
    for block_id in block_table:
        rem = seq_len - len(tokens)
        if rem <= 0:
            break
        take = min(block_size, rem)
        block_tokens = allocator.physical_tokens[block_id, :take]
        tokens.extend(block_tokens.tolist())
    return tokens
