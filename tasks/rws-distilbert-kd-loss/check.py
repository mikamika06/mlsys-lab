import numpy as np
from mlsys import scorers

def _oracle(teacher_logits, student_logits, labels, alpha, temperature):
    eps = 1e-12
    # Temperature‑scaled softmax for teacher
    t_max = np.max(teacher_logits / temperature, axis=1, keepdims=True)
    pt = np.exp((teacher_logits / temperature) - t_max)
    pt /= np.sum(pt, axis=1, keepdims=True)

    # Temperature‑scaled softmax for student
    s_max = np.max(student_logits / temperature, axis=1, keepdims=True)
    ps = np.exp((student_logits / temperature) - s_max)
    ps /= np.sum(ps, axis=1, keepdims=True)

    # KL divergence (mean over batch)
    kl = np.mean(np.sum(pt * (np.log(pt + eps) - np.log(ps + eps)), axis=1))

    # Cross‑entropy with true labels
    ce = -np.mean(np.log(ps[np.arange(labels.size), labels] + eps))

    return alpha * temperature**2 * kl + (1.0 - alpha) * ce

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_rel_err = 0.0
    for _ in range(5):
        N = rng.integers(3, 10)
        C = rng.integers(2, 8)
        teacher_logits = rng.standard_normal((N, C))
        student_logits = rng.standard_normal((N, C))
        labels = rng.integers(0, C, size=N)
        alpha = rng.random()
        temperature = rng.uniform(1.0, 5.0)

        try:
            got = sol.kd_loss(
                teacher_logits,
                student_logits,
                labels,
                alpha=alpha,
                temperature=temperature
            )
        except Exception:
            return {"rel_err": 1e6}

        ref = _oracle(teacher_logits, student_logits, labels, alpha, temperature)
        rel_err = scorers.rel_err(np.array([ref]), np.array([got]))
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": max_rel_err}
