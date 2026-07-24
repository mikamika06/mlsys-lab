import numpy as np
from mlsys import scorers

def _ref(teacher_logits, student_logits):
    """Independent oracle: hard-target CE via manual log-sum-exp."""
    T = np.asarray(teacher_logits, dtype=np.float64)
    S = np.asarray(student_logits, dtype=np.float64)

    # hard targets from teacher
    y = np.argmax(T, axis=-1)                       # (n,)

    # numerically stable log-softmax of student
    m = np.max(S, axis=-1, keepdims=True)           # (n, 1)
    log_sum_exp = m + np.log(np.sum(np.exp(S - m), axis=-1, keepdims=True))
    log_probs = S - log_sum_exp                     # (n, V)

    n = S.shape[0]
    ce = -np.mean(log_probs[np.arange(n), y])
    return float(ce)

def grade(sol, fx) -> dict:
    # Fixed-seed reproducible test inputs (not expected outputs)
    rng = np.random.RandomState(42)

    cases = [
        (rng.randn(32, 100),   rng.randn(32, 100)),
        (rng.randn(16, 500),   rng.randn(16, 500)),
        (rng.randn(64, 2000),  rng.randn(64, 2000)),
        (np.ones((8, 50)),     np.zeros((8, 50))),
        (rng.randn(1, 100),    rng.randn(1, 100)),
    ]

    max_err = 0.0
    for teacher, student in cases:
        ref = _ref(teacher, student)
        try:
            got = float(sol.seq_level_kd_hard(teacher, student))
        except Exception:
            return {"rel_err": 1.0}

        err = scorers.rel_err(np.array([ref]), np.array([got]))
        max_err = max(max_err, err)

    return {"rel_err": max_err}
