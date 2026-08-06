def longest_surviving_prefix(cached_prompts: list[list[int]], new_prompt: list[int]) -> int:
    best = 0
    for cp in cached_prompts:
        match_len = 0
        for ct, nt in zip(cp, new_prompt):
            if ct == nt:
                match_len += 1
            else:
                break
        if match_len > best:
            best = match_len
    return best

def surviving_blocks(
    new_prompt: list[int],
    cached_seqs: list[list[int]],
    block_contents: dict[int, list[int]]
) -> list[int]:
    best_blocks = []
    for seq in cached_seqs:
        current_blocks = []
        token_offset = 0
        for block_id in seq:
            block_tokens = block_contents[block_id]
            b_len = len(block_tokens)
            if token_offset + b_len > len(new_prompt):
                break
            if new_prompt[token_offset : token_offset + b_len] == block_tokens:
                current_blocks.append(block_id)
                token_offset += b_len
            else:
                break
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
    return best_blocks
