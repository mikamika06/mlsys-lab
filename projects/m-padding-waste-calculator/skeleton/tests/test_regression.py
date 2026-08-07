"""Regression tests for sequence packing and boundary invariants."""


def test_bin_capacity_not_exceeded():
    """Verify no bin exceeds max capacity."""
    raise NotImplementedError


def test_cu_seqlens_monotonic():
    """Verify cu_seqlens is monotonically increasing starting at 0."""
    raise NotImplementedError


def test_padding_waste_bounds():
    """Verify waste ratio lies strictly in [0, 1)."""
    raise NotImplementedError
