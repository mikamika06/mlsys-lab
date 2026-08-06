import numpy as np

def fit_student_low_rank(teacher_logits, corpus, rank):
    """
    Fit a student model on the unique states visited in `corpus`.
    Returns a matrix of the same shape as `teacher_logits`.
    For unvisited rows, the student predicts 0.
    For visited rows, the student predicts the rank-`rank` SVD approximation
    of those visited rows from the teacher.
    """
    raise NotImplementedError

def compare_distillation(teacher_logits, synthetic_corpus, real_corpus, rank):
    """
    Compute the MSE between the student and teacher logits when the student
    is fitted on the `synthetic_corpus`, and the MSE when fitted on `real_corpus`.
    Returns (mse_synthetic, mse_real, accuracy_delta) where
    accuracy_delta = mse_synthetic - mse_real.
    The MSE is calculated across ALL entries of the matrix.
    """
    raise NotImplementedError
