"""Regression tests for sequence packing and boundary invariants."""

import sys
sys.path.insert(0, ".")

from varpack.padding import compute_padding_waste
from varpack.packing import pack_sequences_ffd
from varpack.offsets import build_cu_seqlens


def test_bin_capacity_not_exceeded():
    """Verify no bin exceeds max capacity."""
    seq_lens = [120, 450, 300, 800, 250, 600, 100]
    cap = 1000
    bins = pack_sequences_ffd(seq_lens, cap)
    for b in bins:
        assert sum(b["lengths"]) <= cap, f"Bin exceeded capacity: {sum(b['lengths'])} > {cap}"


def test_cu_seqlens_monotonic():
    """Verify cu_seqlens is monotonically increasing starting at 0."""
    seq_lens = [50, 100, 200]
    bins = pack_sequences_ffd(seq_lens, 500)
    cu = build_cu_seqlens(bins)
    assert cu[0] == 0, f"cu_seqlens must start at 0, got {cu[0]}"
    assert len(cu) == len(seq_lens) + 1, f"Expected length {len(seq_lens) + 1}, got {len(cu)}"
    for i in range(len(cu) - 1):
        assert cu[i + 1] > cu[i], f"Non-monotonic at index {i}: {cu[i]} >= {cu[i+1]}"


def test_padding_waste_bounds():
    """Verify waste ratio lies strictly in [0, 1)."""
    seq_lens = [100, 200, 300]
    waste = compute_padding_waste(seq_lens)
    assert 0.0 <= waste < 1.0, f"Waste ratio out of bounds: {waste}"
