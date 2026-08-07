import sys

sys.path.insert(0, ".")
from swapspace.sizing import compute_sequence_swap_bytes, compute_total_swap_bytes

CONFIG = {
    "num_layers": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "dtype_bytes": 2,
    "block_size": 16,
}


def test_block_padding_underallocation():
    unaligned_tokens = 17
    seq_bytes = compute_sequence_swap_bytes(CONFIG, unaligned_tokens)
    exact_unpadded = (
        2
        * CONFIG["num_layers"]
        * CONFIG["num_kv_heads"]
        * CONFIG["head_dim"]
        * unaligned_tokens
        * CONFIG["dtype_bytes"]
    )
    assert seq_bytes > exact_unpadded, "Swap space allocation failed to account for block boundary padding"


def test_total_swap_bytes_consistency():
    tokens = [15, 16, 17]
    tot = compute_total_swap_bytes(CONFIG, tokens)
    indiv_sum = sum(compute_sequence_swap_bytes(CONFIG, t) for t in tokens)
    assert tot == indiv_sum, f"Total swap bytes {tot} does not match sum of individual sequence bytes {indiv_sum}"
