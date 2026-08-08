import sys
import numpy as np

sys.path.insert(0, ".")

from fsdp_ckpt.converter import consolidate, shard_checkpoint
from fsdp_ckpt.parser import extract_chunks, align_shapes


def test_shard_consolidate_invariant():
    """Verify that sharding to a new number of ranks and recovering yields exact same arrays."""
    metadata = {"layer1.weight": (13, 17), "layer1.bias": (19,)}
    consolidated = {
        "layer1.weight": np.ones((13, 17)),
        "layer1.bias": np.arange(19, dtype=float)
    }

    sharded = shard_checkpoint(consolidated, 7)
    chunks = extract_chunks(sharded)
    aligned = align_shapes(chunks, metadata)
    reconstructed = consolidate(aligned, metadata)

    assert np.array_equal(reconstructed["layer1.weight"], consolidated["layer1.weight"])
    assert np.array_equal(reconstructed["layer1.bias"], consolidated["layer1.bias"])
