"""Cumulative sequence length offsets for varlen execution."""

import numpy as np


def build_cu_seqlens(packed_bins):
    """Construct prefix sum array (cu_seqlens) for varlen attention kernel."""
    raise NotImplementedError


def build_sequence_metadata(packed_bins):
    """Extract flat token counts, max sequence length, and total sequence count."""
    raise NotImplementedError
