def build_paged_split_plan(block_table: list[int], page_size: int, num_splits: int) -> list[dict]:
    total_blocks = len(block_table)
    if total_blocks == 0 or num_splits <= 0:
        return []
    actual_splits = min(num_splits, total_blocks)
    blocks_per_split = total_blocks // actual_splits
    remainder = total_blocks % actual_splits

    plan = []
    start_blk = 0
    start_tok = 0

    for i in range(actual_splits):
        count = blocks_per_split + (1 if i < remainder else 0)
        end_blk = start_blk + count
        end_tok = end_blk * page_size
        sub_blocks = block_table[start_blk:end_blk]

        plan.append({
            "split_id": i,
            "start_token": start_tok,
            "end_token": end_tok,
            "blocks": sub_blocks,
        })
        start_blk = end_blk
        start_tok = end_tok

    return plan
