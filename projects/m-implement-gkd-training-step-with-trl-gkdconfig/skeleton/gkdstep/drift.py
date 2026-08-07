import numpy as np


def compute_tv_distance(p_probs, q_probs):
    """Compute total variation distance between probability distributions."""
    raise NotImplementedError


def measure_sequence_drift(teacher_seq_logits, student_seq_logits, beta=0.5, temperature=1.0):
    """Measure cumulative distribution drift over autoregressive time steps."""
    raise NotImplementedError
