from layout.derivation import derive_block_size

def test_block_size_avx512():
    assert derive_block_size("avx512", 64) == 16
