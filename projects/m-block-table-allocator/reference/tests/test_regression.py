import numpy as np
from pagedkv.allocator import BlockAllocator
from pagedkv.metrics import compute_fragmentation


def test_fragmentation_metric_accuracy():
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    allocator.allocate("seq1", 20)
    seq_lengths = {"seq1": 20}
    metrics = compute_fragmentation(allocator, seq_lengths)

    expected_internal = (32 - 20) / 32
    expected_external = (8 * 16) / 160

    assert np.isclose(metrics["internal_fragmentation"], expected_internal)
    assert np.isclose(metrics["external_fragmentation"], expected_external)
