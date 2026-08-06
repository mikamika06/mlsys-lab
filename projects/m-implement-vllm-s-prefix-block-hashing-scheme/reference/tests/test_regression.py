import sys

sys.path.insert(0, ".")
from prefix_cache.block_hash import compute_prefix_hashes


def test_block_hash_chain():
    seq_a = list(range(32))
    seq_b = list(range(32))
    seq_b[0] = 999

    hashes_a = compute_prefix_hashes(seq_a, block_size=16)
    hashes_b = compute_prefix_hashes(seq_b, block_size=16)

    assert len(hashes_a) == 2
    assert len(hashes_b) == 2
    assert hashes_a[0] != hashes_b[0]
    assert hashes_a[1] != hashes_b[1]
