import numpy as np

def generate_fixtures():
    rng = np.random.RandomState(123)
    V, d = 20, 20
    teacher_logits = rng.randn(V, d) * 2.0

    real_corpus = rng.randint(0, V, size=100)
    return teacher_logits, real_corpus

def get_transition_probs(logits):
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)

def sample_teacher(logits, start_state, steps, seed=42):
    rng = np.random.RandomState(seed)
    probs = get_transition_probs(logits)
    V = logits.shape[0]
    seq = [start_state]
    curr = start_state
    for _ in range(steps - 1):
        curr = rng.choice(V, p=probs[curr])
        seq.append(curr)
    return np.array(seq)

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

def min_diversity_bound(teacher_logits, rank, target_mse):
    V, d = teacher_logits.shape
    norms = np.sum(teacher_logits ** 2, axis=1)
    sorted_idx = np.argsort(-norms)
    for k in range(1, V + 1):
        visited = sorted_idx[:k]
        T_sub = teacher_logits[visited]
        U, S, Vh = np.linalg.svd(T_sub, full_matrices=False)
        r = min(rank, len(S))
        trunc_err = np.sum(S[r:] ** 2)
        unvis_err = np.sum(norms[sorted_idx[k:]])
        if (trunc_err + unvis_err) / (V * d) <= target_mse:
            return k
    return V
