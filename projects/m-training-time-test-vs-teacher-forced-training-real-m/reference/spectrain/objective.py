import numpy as np
from spectrain.acceptance import compute_acceptance_rate
from spectrain.teacher import teacher_forced_loss


def combined_objective(tokens, draft_logits, target_logits, gamma=4, alpha=0.5):
    loss = teacher_forced_loss(tokens, draft_logits)
    acc = compute_acceptance_rate(tokens, draft_logits, target_logits, gamma)
    return float(loss - alpha * acc)
