import math


def predict_latency(batch_size: int, seq_len: int, num_splits: int, num_sm: int = 108) -> float:
    total_ctas = batch_size * num_splits
    waves = math.ceil(total_ctas / float(num_sm))
    tokens_per_split = math.ceil(seq_len / float(num_splits))
    compute_cost = 0.05 * tokens_per_split
    reduction_overhead = 0.0 if num_splits == 1 else (12.0 + 3.5 * num_splits)
    return float(waves * (compute_cost + reduction_overhead))


def optimal_num_splits(batch_size: int, seq_len: int, num_sm: int = 108) -> int:
    best_s = 1
    best_lat = float("inf")
    for s in range(1, 65):
        lat = predict_latency(batch_size, seq_len, s, num_sm)
        if lat < best_lat:
            best_lat = lat
            best_s = s
    return best_s


def find_crossover_batch(seq_len: int, num_sm: int = 108) -> int:
    for b in range(1, 512):
        if optimal_num_splits(b, seq_len, num_sm) == 1:
            return b
    return 512


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


CONFIGS = [
    {"batch_size": 1, "seq_len": 4096, "num_sm": 108},
    {"batch_size": 4, "seq_len": 8192, "num_sm": 108},
    {"batch_size": 16, "seq_len": 16384, "num_sm": 108},
    {"batch_size": 64, "seq_len": 32768, "num_sm": 108},
    {"batch_size": 128, "seq_len": 8192, "num_sm": 108},
]

CROSSOVER_SEQS = [2048, 4096, 8192, 16384, 32768]
