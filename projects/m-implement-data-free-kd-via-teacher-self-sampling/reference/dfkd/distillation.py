import numpy as np

def fit_student(sequence, teacher_logits, rank):
    V = teacher_logits.shape[0]
    visited = np.zeros(V, dtype=bool)
    visited[sequence] = True

    student_logits = np.zeros_like(teacher_logits)
    if not np.any(visited):
        return student_logits

    T_sub = teacher_logits[visited]
    U, S, Vh = np.linalg.svd(T_sub, full_matrices=False)

    r = min(rank, len(S))
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    Vh_r = Vh[:r, :]

    student_logits[visited] = U_r @ S_r @ Vh_r
    return student_logits

def evaluate_mse(student_logits, teacher_logits):
    return np.mean((student_logits - teacher_logits) ** 2)
