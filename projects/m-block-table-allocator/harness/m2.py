import numpy as np


def check(workdir):
    try:
        from pagedkv.allocator import BlockAllocator
        from pagedkv.gather import gather_kv_cache
        from pagedkv.metrics import compute_fragmentation
    except Exception as e:
        return {"gather_matched": 0.0, "fragmentation_matched": 0.0, "_note": f"Import failed: {e}"}

    np.random.seed(42)
    num_blocks, block_size, num_heads, head_dim = 8, 16, 4, 32
    physical = np.random.randn(num_blocks, block_size, num_heads, head_dim).astype(np.float32)

    allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    allocator.allocate("seq1", 25)
    allocator.allocate("seq2", 40)

    bt1 = allocator.get_block_table("seq1")
    gathered1 = gather_kv_cache(physical, bt1, 25)

    expected1 = np.concatenate([physical[b] for b in bt1], axis=0)[:25]
    if not np.allclose(gathered1, expected1):
        return {"gather_matched": 0.0, "fragmentation_matched": 0.0, "_note": "Gather output does not match physical storage slice"}

    seq_lengths = {"seq1": 25, "seq2": 40}
    metrics = compute_fragmentation(allocator, seq_lengths)

    expected_internal = ((32 - 25) + (48 - 40)) / (32 + 48)
    expected_external = (3 * 16) / (8 * 16)

    if not np.isclose(metrics["internal_fragmentation"], expected_internal) or not np.isclose(metrics["external_fragmentation"], expected_external):
        return {"gather_matched": 1.0, "fragmentation_matched": 0.0, "_note": f"Fragmentation metrics mismatch: {metrics}"}

    return {"gather_matched": 1.0, "fragmentation_matched": 1.0}
