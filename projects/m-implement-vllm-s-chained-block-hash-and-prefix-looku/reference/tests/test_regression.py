import sys

sys.path.insert(0, ".")
from prefix_cache.hash import build_prefix_hash_chain, compute_block_hash


def test_block_hash_chaining_dependency():
    tokens_a = [10, 20, 30, 40]
    tokens_b = [10, 20, 30, 40]
    chain_a = build_prefix_hash_chain(tokens_a, block_size=2)
    chain_b = build_prefix_hash_chain(tokens_b, block_size=2)
    assert chain_a == chain_b

    tokens_c = [99, 88, 30, 40]
    chain_c = build_prefix_hash_chain(tokens_c, block_size=2)
    assert (
        chain_a[1] != chain_c[1]
    ), "Second block hash must differ when parent block differs, even with identical block tokens"


def test_compute_block_hash_parent_influence():
    block = [1, 2, 3, 4]
    h1 = compute_block_hash(block, parent_hash=None)
    h2 = compute_block_hash(block, parent_hash="some_parent_hash")
    assert (
        h1 != h2
    ), "Block hash without parent must differ from block hash with parent"
