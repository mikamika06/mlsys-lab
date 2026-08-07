import numpy as np


def run_epoch(student_logits, teacher_logits, targets, alpha, temperature, mode):
    raise NotImplementedError


def evaluate_accuracy(logits, targets):
    raise NotImplementedError
