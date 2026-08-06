import sys

sys.path.insert(0, ".")
from simulator.core import append_tokens, decode_bandwidth

def test_append_tokens_no_off_by_one():
    seqlens = [15, 16, 17]
    tables = [[100], [101, 102], [103, 104]]
    block_size = 16

    got = append_tokens(seqlens, tables, block_size)

    assert got[0] == (100, 15), f"Expected block 100 offset 15, got {got[0]}"
    assert got[1] == (102, 0), f"Expected block 102 offset 0, got {got[1]}"
    assert got[2] == (104, 1), f"Expected block 104 offset 1, got {got[2]}"

def test_decode_bandwidth_scaling():
    b1 = decode_bandwidth([10, 10], 1, 8, 128, 2)
    b2 = decode_bandwidth([10, 10], 2, 8, 128, 2)
    assert b2 == b1 * 2, "Bandwidth should scale linearly with layers"
