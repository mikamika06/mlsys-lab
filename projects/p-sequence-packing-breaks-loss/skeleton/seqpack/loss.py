import numpy as np


def compute_naive_packed_loss(logits, labels, label_mask):
    """Computes naive packed loss ignoring sequence boundary constraints."""
    raise NotImplementedError


def compute_packed_loss(logits, labels, label_mask, seq_ids):
    """Computes token cross-entropy loss correctly normalized across valid targets."""
    raise NotImplementedError
