import numpy as np


def softmax(logits, temperature=1.0):
    """Compute temperature-scaled softmax probabilities."""
    raise NotImplementedError


def compute_divergence(teacher_logits, student_logits, divergence_type="forward_kl", temperature=1.0):
    """Compute divergence loss between teacher and student logits."""
    raise NotImplementedError


def compute_gkd_step_loss(teacher_logits, student_logits, config):
    """Compute GKD step loss using GKDConfig parameters."""
    raise NotImplementedError
