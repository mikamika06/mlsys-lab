import numpy as np


def estimate_activation_memory_and_comm(seq_len, batch_size, hidden_dim, tp_size, bytes_per_elem=2):
    """Estimate per-rank activation memory and communication volume for TP-only vs TP+SP."""
    raise NotImplementedError
