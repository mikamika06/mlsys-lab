import numpy as np
from kd.loss import blended_loss, hard_loss, soft_loss


def run_epoch(student_logits, teacher_logits, targets, alpha, temperature, mode):
    if mode == "hard":
        return hard_loss(student_logits, targets)
    elif mode == "soft":
        return soft_loss(student_logits, teacher_logits, temperature)
    elif mode == "blended":
        return blended_loss(student_logits, targets, student_logits, teacher_logits, alpha, temperature)
    else:
        raise ValueError(f"unknown mode {mode}")


def evaluate_accuracy(logits, targets):
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == targets))
