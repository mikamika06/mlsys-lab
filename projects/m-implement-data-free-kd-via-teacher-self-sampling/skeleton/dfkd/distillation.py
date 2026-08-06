import numpy as np

def fit_student(sequence, teacher_logits, rank):
    """
    Fit a student model on the states visited in `sequence`.
    Returns a matrix of the same shape as `teacher_logits`.
    For unvisited rows, the student predicts 0.
    For visited rows, the student predicts the rank-`rank` SVD approximation
    of those visited rows from the teacher.
    """
    raise NotImplementedError

def evaluate_mse(student_logits, teacher_logits):
    """
    Compute the Mean Squared Error across ALL entries of the logits matrix.
    """
    raise NotImplementedError
