"""Padding waste and efficiency calculations."""

import numpy as np


def compute_padding_waste(seq_lens, max_len=None):
    """Calculate ratio of padding tokens relative to total padded tokens."""
    raise NotImplementedError


def compute_flop_savings(seq_lens):
    """Calculate relative FLOP savings when switching from square batch to varlen attention."""
    raise NotImplementedError
