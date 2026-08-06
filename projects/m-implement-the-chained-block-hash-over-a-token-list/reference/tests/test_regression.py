import sys

sys.path.insert(0, ".")
from blockhash.hashing import block_hashes

def test_chained_hash_prevents_middle_sharing():
    t1 = [1, 2, 3, 4, 5, 6, 7, 8]
    t2 = [9, 9, 3, 4, 5, 6, 7, 8]
    h1 = block_hashes(t1, 4)
    h2 = block_hashes(t2, 4)

    assert h1[1] != h2[1], "Block 1 hashes should differ because the prefixes diverged in Block 0"
