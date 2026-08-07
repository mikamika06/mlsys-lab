"""Cumulative sequence length offsets for varlen execution."""

import numpy as np


def build_cu_seqlens(packed_bins):
    """Construct prefix sum array (cu_seqlens) for varlen attention kernel."""
    cu = [0]
    curr = 0
    for b in packed_bins:
        for l in b["lengths"]:
            curr += l
            cu.append(curr)
    return np.array(cu, dtype=np.int32)


def build_sequence_metadata(packed_bins):
    """Extract flat token counts, max sequence length, and total sequence count."""
    all_lens = [l for b in packed_bins for l in b["lengths"]]
    if not all_lens:
        return {"total_tokens": 0, "max_seqlen": 0, "num_sequences": 0}
    return {
        "total_tokens": int(sum(all_lens)),
        "max_seqlen": int(max(all_lens)),
        "num_sequences": int(len(all_lens)),
    }
