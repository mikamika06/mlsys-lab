import numpy as np


def hard_loss(logits, targets):
    raise NotImplementedError


def soft_loss(student_logits, teacher_logits, temperature):
    raise NotImplementedError


def blended_loss(logits, targets, student_logits, teacher_logits, alpha, temperature):
    raise NotImplementedError
