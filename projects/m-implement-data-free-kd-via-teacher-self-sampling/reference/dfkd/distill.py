import numpy as np

def fit_student_low_rank(teacher_logits, corpus, rank):
    V, d = teacher_logits.shape
    student = np.zeros_like(teacher_logits)
    visited = np.unique(corpus)
    if len(visited) == 0:
        return student

    T_sub = teacher_logits[visited]
    U, S, Vh = np.linalg.svd(T_sub, full_matrices=False)

    r = min(rank, len(S))
    student[visited] = (U[:, :r] * S[:r]) @ Vh[:r, :]
    return student

def compare_distillation(teacher_logits, synthetic_corpus, real_corpus, rank):
    s_syn = fit_student_low_rank(teacher_logits, synthetic_corpus, rank)
    s_real = fit_student_low_rank(teacher_logits, real_corpus, rank)

    mse_syn = np.mean((s_syn - teacher_logits) ** 2)
    mse_real = np.mean((s_real - teacher_logits) ** 2)

    return mse_syn, mse_real, mse_syn - mse_real
